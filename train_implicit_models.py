"""
Train ALS, BPR (via implicit library) and SVD models.
Saves trained models to ./models/ directory.
"""
import os
import pickle

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")
import django
django.setup()

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import implicit

from analytics.models import Rating


def check_gpu():
    try:
        import implicit.gpu
        # Actually test if CUDA works
        implicit.gpu.als.AlternatingLeastSquares(factors=2)
        return True
    except Exception:
        return False


def _build_matrices(df):
    """Build sparse matrices and mappings from a ratings DataFrame."""
    df = df.copy()
    df['user_id'] = df['user_id'].astype('category')
    df['movie_id'] = df['movie_id'].astype('category')

    user_map = dict(enumerate(df['user_id'].cat.categories))
    item_map = dict(enumerate(df['movie_id'].cat.categories))
    user_to_idx = {v: k for k, v in user_map.items()}
    item_to_idx = {v: k for k, v in item_map.items()}

    sparse_item_user = coo_matrix(
        (df['rating'].astype(float),
         (df['movie_id'].cat.codes, df['user_id'].cat.codes))
    ).tocsr()

    sparse_user_item = coo_matrix(
        (df['rating'].astype(float),
         (df['user_id'].cat.codes, df['movie_id'].cat.codes))
    ).tocsr()

    return sparse_item_user, sparse_user_item, user_map, item_map, user_to_idx, item_to_idx


def load_ratings():
    print("Loading ratings from DB...")
    data = list(Rating.objects.all().values('user_id', 'movie_id', 'rating'))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} ratings")
    return _build_matrices(df)


def load_ratings_from_split(split_path):
    """Load only TRAIN ratings from a saved split file."""
    import json
    with open(split_path) as f:
        raw = json.load(f)

    rows = []
    for user_id, data in raw.items():
        for item_id, rating in data['train_ratings'].items():
            rows.append({'user_id': user_id, 'movie_id': item_id, 'rating': float(rating)})

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} TRAIN ratings from split ({len(raw)} users)")
    return _build_matrices(df)


def train_als(sparse_user_item, save_dir, use_gpu=False):
    print("\n=== Training ALS ===")
    os.makedirs(save_dir, exist_ok=True)

    model = implicit.als.AlternatingLeastSquares(
        factors=128,
        regularization=0.01,
        iterations=30,
        use_gpu=use_gpu,
    )
    # Use binary interactions (item was rated) with mild confidence boost
    confidence = sparse_user_item.copy()
    confidence.data = np.ones_like(confidence.data, dtype=np.float32)
    model.fit(confidence)

    # Convert GPU model to CPU for serialization
    if use_gpu:
        model = model.to_cpu()

    with open(os.path.join(save_dir, 'als_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"ALS model saved to {save_dir}")
    return model


def train_bpr(sparse_user_item, save_dir, use_gpu=False):
    print("\n=== Training BPR ===")
    os.makedirs(save_dir, exist_ok=True)

    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=128,
        learning_rate=0.03,
        regularization=0.005,
        iterations=100,
        use_gpu=use_gpu,
    )
    binary = (sparse_user_item > 0).astype(np.float32)
    model.fit(binary)

    if use_gpu:
        model = model.to_cpu()

    with open(os.path.join(save_dir, 'bpr_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"BPR model saved to {save_dir}")
    return model


def train_svd(sparse_user_item, save_dir, k=200):
    """
    SVD for ranking: apply truncated SVD to log-scaled interaction matrix.
    Uses log(1 + rating) to dampen extreme values, then TF-IDF-style
    normalization to reduce popularity bias.

    Predictions: score(u,i) = U_sigma[u] @ Vt[:, i]  (pure latent ranking)
    """
    print("\n=== Training SVD ===")
    os.makedirs(save_dir, exist_ok=True)

    from scipy.sparse import csr_matrix

    mat = csr_matrix(sparse_user_item.astype(float).copy())

    # Boost high ratings: square ratings so 5-star items stand out more
    mat.data = mat.data ** 2

    # Log-scale to dampen extreme magnitude differences
    mat.data = np.log1p(mat.data)

    # Row-normalize (L2) so active users don't dominate
    from sklearn.preprocessing import normalize
    mat = normalize(mat, norm='l2', axis=1)

    k = min(k, min(mat.shape) - 1)
    U, sigma, Vt = svds(mat, k=k)

    # Sort by descending singular value
    idx = np.argsort(-sigma)
    U = U[:, idx]
    sigma = sigma[idx]
    Vt = Vt[idx, :]

    np.save(os.path.join(save_dir, 'U.npy'), U)
    np.save(os.path.join(save_dir, 'sigma.npy'), sigma)
    np.save(os.path.join(save_dir, 'Vt.npy'), Vt)

    print(f"SVD model saved to {save_dir} (k={U.shape[1]}, top sigma={sigma[0]:.3f})")


def save_mappings(user_map, item_map, user_to_idx, item_to_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'user_map.pkl'), 'wb') as f:
        pickle.dump(user_map, f)
    with open(os.path.join(save_dir, 'item_map.pkl'), 'wb') as f:
        pickle.dump(item_map, f)
    with open(os.path.join(save_dir, 'user_to_idx.pkl'), 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open(os.path.join(save_dir, 'item_to_idx.pkl'), 'wb') as f:
        pickle.dump(item_to_idx, f)
    print(f"Mappings saved ({len(user_map)} users, {len(item_map)} items)")


if __name__ == '__main__':
    use_gpu = check_gpu()
    print(f"GPU available: {use_gpu}")

    sparse_item_user, sparse_user_item, user_map, item_map, user_to_idx, item_to_idx = load_ratings()

    base_dir = './models'
    save_mappings(user_map, item_map, user_to_idx, item_to_idx, base_dir)

    train_als(sparse_user_item, os.path.join(base_dir, 'als'), use_gpu=use_gpu)
    train_bpr(sparse_user_item, os.path.join(base_dir, 'bpr'), use_gpu=use_gpu)
    train_svd(sparse_user_item, os.path.join(base_dir, 'svd'))

    print("\n=== All models trained! ===")
