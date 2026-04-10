"""
Evaluate all recommendation models using standard IR metrics.

Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage.

Methodology:
  For each test user, hold out 20% of ratings as ground truth.
  Ask each model for top-K from remaining items.
  A recommendation is "relevant" if in held-out set with rating >= 4.0.

IMPORTANT: Models must exclude only TRAIN items (not all DB ratings),
otherwise test items get filtered out and metrics are always 0.
"""
import os
import sys
import time
import random
import argparse
from collections import defaultdict

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django
django.setup()

import numpy as np
import pandas as pd
from django.db.models import Count, Avg

from analytics.models import Rating
from recommender.models import Similarity
from school_items.models import Item


# -- Metrics ---------------------------------------------------------------

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & set(relevant)) / len(rec_k)


def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(rec_k) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    if not relevant:
        return 0.0
    return sum_precision / min(len(relevant), k)


def hit_rate_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return 1.0 if set(rec_k) & set(relevant) else 0.0


# -- Model wrappers (train-aware: exclude only train_items) ----------------

def get_item_based_cf_recs(train_items, num=20):
    """Item-based CF from Similarity table."""
    train_set = set(train_items)
    sims = Similarity.objects.filter(source__in=train_items) \
        .exclude(target__in=train_items) \
        .values('target') \
        .annotate(score=Count('target')) \
        .order_by('-score')[:num]
    return [s['target'] for s in sims]


def get_content_based_recs(user_id, train_items, train_ratings_dict, num=20):
    """Content-based via TF-IDF (LdaSimilarity): weighted by user ratings."""
    from django.db.models import Q
    from recommender.models import LdaSimilarity

    if not train_items:
        return []

    train_set = set(train_items)
    user_mean = float(sum(float(v) for v in train_ratings_dict.values()) / len(train_ratings_dict)) if train_ratings_dict else 3.0

    # Get TF-IDF similarities for items in user's train set
    sims = LdaSimilarity.objects.filter(
        Q(source__in=train_items) & ~Q(target__in=train_items) & Q(similarity__gt=0.1)
    ).order_by('-similarity')[:200]

    recs = {}
    for s in sims:
        target = s.target
        source_rating = float(train_ratings_dict.get(s.source, user_mean))
        weight = float(s.similarity) * (source_rating - user_mean)

        if target not in recs:
            recs[target] = {'score': 0.0, 'sim_sum': 0.0}
        recs[target]['score'] += weight
        recs[target]['sim_sum'] += float(s.similarity)

    result = []
    for target, data in recs.items():
        if data['sim_sum'] > 0:
            pred = user_mean + data['score'] / data['sim_sum']
            result.append((target, pred))

    result.sort(key=lambda x: -x[1])
    return [r[0] for r in result[:num]]


def get_popularity_recs(train_items, num=20):
    """Popularity from the same item category, excluding train items."""
    item = Item.objects.filter(item_id=train_items[0]).first() if train_items else None
    if not item or not item.primary_category:
        return []

    same_cat = Item.objects.filter(categories_en__icontains=item.primary_category) \
        .exclude(item_id__in=train_items) \
        .values_list('item_id', flat=True)

    pop = Rating.objects.filter(movie_id__in=same_cat) \
        .values('movie_id').annotate(cnt=Count('user_id')).order_by('-cnt')[:num]
    return [p['movie_id'] for p in pop]


# -- Matrix factorization models (compute scores directly, exclude only train) --

def _load_mappings():
    """Load user/item index mappings (cached)."""
    if not hasattr(_load_mappings, '_cache'):
        import pickle
        model_dir = './models'
        try:
            with open(os.path.join(model_dir, 'user_to_idx.pkl'), 'rb') as f:
                user_to_idx = pickle.load(f)
            with open(os.path.join(model_dir, 'item_map.pkl'), 'rb') as f:
                item_map = pickle.load(f)
            with open(os.path.join(model_dir, 'item_to_idx.pkl'), 'rb') as f:
                item_to_idx = pickle.load(f)
            _load_mappings._cache = (user_to_idx, item_map, item_to_idx)
        except FileNotFoundError:
            _load_mappings._cache = (None, None, None)
    return _load_mappings._cache


_cached_models = {}


def _get_model(name):
    if name not in _cached_models:
        import pickle
        model_dir = './models'
        try:
            if name == 'als':
                with open(os.path.join(model_dir, 'als', 'als_model.pkl'), 'rb') as f:
                    _cached_models[name] = pickle.load(f)
                print(f"  Loaded ALS model")
            elif name == 'bpr':
                with open(os.path.join(model_dir, 'bpr', 'bpr_model.pkl'), 'rb') as f:
                    _cached_models[name] = pickle.load(f)
                print(f"  Loaded BPR model")
            elif name == 'svd':
                svd_dir = os.path.join(model_dir, 'svd')
                U = np.load(os.path.join(svd_dir, 'U.npy'))
                sigma = np.load(os.path.join(svd_dir, 'sigma.npy'))
                Vt = np.load(os.path.join(svd_dir, 'Vt.npy'))
                mean = np.load(os.path.join(svd_dir, 'user_ratings_mean.npy'))
                predicted = U @ np.diag(sigma) @ Vt
                _cached_models[name] = (predicted, mean)
                print(f"  Loaded SVD model")
            elif name == 'cf':
                from recs.neighborhood_based_recommender import NeighborhoodBasedRecs
                _cached_models[name] = NeighborhoodBasedRecs(min_sim=0.0)
                print(f"  Loaded Neighborhood CF")
        except Exception as e:
            print(f"  Warning: could not load {name}: {e}")
            _cached_models[name] = None
    return _cached_models.get(name)


def get_als_recs(user_id, train_items, num=20):
    """ALS: compute scores from factors, exclude only train_items."""
    model = _get_model('als')
    user_to_idx, item_map, item_to_idx = _load_mappings()
    if model is None or user_to_idx is None or user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]
    user_vec = model.user_factors[user_idx]
    scores = model.item_factors @ user_vec

    # Exclude only train items
    train_indices = set(item_to_idx[iid] for iid in train_items if iid in item_to_idx)
    for idx in train_indices:
        scores[idx] = -999

    top_idx = np.argsort(-scores)[:num]
    return [item_map[idx] for idx in top_idx if idx in item_map]


def get_bpr_recs(user_id, train_items, num=20):
    """BPR: compute scores from factors, exclude only train_items."""
    model = _get_model('bpr')
    user_to_idx, item_map, item_to_idx = _load_mappings()
    if model is None or user_to_idx is None or user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]
    user_vec = model.user_factors[user_idx]
    scores = model.item_factors @ user_vec

    train_indices = set(item_to_idx[iid] for iid in train_items if iid in item_to_idx)
    for idx in train_indices:
        scores[idx] = -999

    top_idx = np.argsort(-scores)[:num]
    return [item_map[idx] for idx in top_idx if idx in item_map]


def get_svd_recs(user_id, train_items, num=20):
    """SVD: use precomputed predicted matrix, exclude only train_items."""
    data = _get_model('svd')
    user_to_idx, item_map, item_to_idx = _load_mappings()
    if data is None or user_to_idx is None or user_id not in user_to_idx:
        return []

    predicted, mean = data
    user_idx = user_to_idx[user_id]
    scores = predicted[user_idx] + mean[user_idx]

    train_indices = set(item_to_idx[iid] for iid in train_items if iid in item_to_idx)
    for idx in train_indices:
        scores[idx] = -999

    top_idx = np.argsort(-scores)[:num]
    return [item_map[idx] for idx in top_idx if idx in item_map]


def get_neighborhood_cf_recs(user_id, train_items, num=20):
    """Neighborhood CF - uses DB ratings internally (known limitation)."""
    m = _get_model('cf')
    if not m:
        return []
    try:
        items = m.recommend_items(user_id, num * 2)
        # Post-filter: keep items NOT in train_items (allow test items through)
        train_set = set(train_items)
        return [item[0] for item in items if item[0] not in train_set][:num]
    except Exception:
        return []


def get_fwls_recs(user_id, train_items, train_ratings_dict, num=20):
    """FWLS hybrid - combine CB + CF scores manually for evaluation."""
    # CB part: TF-IDF based
    cb_recs = get_content_based_recs(user_id, train_items, train_ratings_dict, num * 2)
    # CF part: item-based
    cf_recs = get_item_based_cf_recs(train_items, num * 2)

    # Simple rank fusion: items appearing in both get boosted
    scores = defaultdict(float)
    for rank, item in enumerate(cf_recs):
        scores[item] += 1.0 / (rank + 1)
    for rank, item in enumerate(cb_recs):
        scores[item] += 0.5 / (rank + 1)  # CB weight lower

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_items[:num]]


# -- Main evaluation -------------------------------------------------------

def prepare_test_data(min_ratings=10, test_fraction=0.2, max_users=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    users = list(
        Rating.objects.values('user_id')
        .annotate(cnt=Count('user_id'))
        .filter(cnt__gte=min_ratings)
        .order_by('-cnt')
        .values_list('user_id', flat=True)[:max_users * 3]
    )

    if len(users) > max_users:
        users = random.sample(list(users), max_users)

    print(f"Selected {len(users)} test users (min {min_ratings} ratings each)")

    test_data = {}
    for user_id in users:
        ratings = list(
            Rating.objects.filter(user_id=user_id)
            .values('movie_id', 'rating')
            .order_by('?')
        )
        n_test = max(1, int(len(ratings) * test_fraction))
        test_ratings = ratings[:n_test]
        train_ratings = ratings[n_test:]

        train_items = [r['movie_id'] for r in train_ratings]
        train_ratings_dict = {r['movie_id']: r['rating'] for r in train_ratings}
        test_relevant = [r['movie_id'] for r in test_ratings if r['rating'] >= 4.0]

        if test_relevant:
            test_data[user_id] = (train_items, train_ratings_dict, test_relevant)

    print(f"Users with relevant test items: {len(test_data)}")
    return test_data


def evaluate_model(model_name, rec_fn, test_data, k_values=[5, 10, 20]):
    all_recommended = set()
    metrics = {k: defaultdict(list) for k in k_values}

    t0 = time.time()
    evaluated = 0
    errors = 0

    for user_id, (train_items, train_ratings_dict, test_relevant) in test_data.items():
        try:
            recommended = rec_fn(user_id, train_items, train_ratings_dict, max(k_values))
        except Exception as e:
            errors += 1
            continue

        if not recommended:
            for k in k_values:
                metrics[k]['precision'].append(0.0)
                metrics[k]['recall'].append(0.0)
                metrics[k]['ndcg'].append(0.0)
                metrics[k]['map'].append(0.0)
                metrics[k]['hit_rate'].append(0.0)
            evaluated += 1
            continue

        all_recommended.update(recommended)

        for k in k_values:
            metrics[k]['precision'].append(precision_at_k(recommended, test_relevant, k))
            metrics[k]['recall'].append(recall_at_k(recommended, test_relevant, k))
            metrics[k]['ndcg'].append(ndcg_at_k(recommended, test_relevant, k))
            metrics[k]['map'].append(average_precision_at_k(recommended, test_relevant, k))
            metrics[k]['hit_rate'].append(hit_rate_at_k(recommended, test_relevant, k))

        evaluated += 1

        if evaluated % 50 == 0:
            print(f"    ... {evaluated} users done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    total_items = Item.objects.count()
    coverage = len(all_recommended) / total_items if total_items > 0 else 0

    results = {'model': model_name, 'evaluated': evaluated, 'errors': errors,
               'time_sec': round(elapsed, 2), 'coverage': round(coverage, 4)}

    for k in k_values:
        for metric_name in ['precision', 'recall', 'ndcg', 'map', 'hit_rate']:
            vals = metrics[k][metric_name]
            results[f'{metric_name}@{k}'] = round(np.mean(vals), 4) if vals else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--users', type=int, default=200, help='Max test users')
    parser.add_argument('--min-ratings', type=int, default=10, help='Min ratings per user')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20], help='K values')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("RECOMMENDATION MODELS EVALUATION")
    print("=" * 70)

    test_data = prepare_test_data(
        min_ratings=args.min_ratings,
        max_users=args.users,
        seed=args.seed,
    )

    # All models: fn(user_id, train_items, train_ratings_dict, num) -> [item_id, ...]
    models = {
        'Item-based CF': lambda uid, train, rd, n: get_item_based_cf_recs(train, n),
        'Neighborhood CF': lambda uid, train, rd, n: get_neighborhood_cf_recs(uid, train, n),
        'Content-Based TF-IDF': lambda uid, train, rd, n: get_content_based_recs(uid, train, rd, n),
        'ALS (implicit)': lambda uid, train, rd, n: get_als_recs(uid, train, n),
        'BPR (implicit)': lambda uid, train, rd, n: get_bpr_recs(uid, train, n),
        'SVD (scipy)': lambda uid, train, rd, n: get_svd_recs(uid, train, n),
        'FWLS (hybrid)': lambda uid, train, rd, n: get_fwls_recs(uid, train, rd, n),
        'Popularity': lambda uid, train, rd, n: get_popularity_recs(train, n),
    }

    all_results = []

    for model_name, rec_fn in models.items():
        print(f"\n{'-' * 50}")
        print(f"Evaluating: {model_name}")
        print(f"{'-' * 50}")

        result = evaluate_model(model_name, rec_fn, test_data, args.k)
        all_results.append(result)

        print(f"  Users: {result['evaluated']}, Errors: {result['errors']}, "
              f"Time: {result['time_sec']}s, Coverage: {result['coverage']:.2%}")
        for k in args.k:
            print(f"  @{k}: P={result[f'precision@{k}']:.4f}  "
                  f"R={result[f'recall@{k}']:.4f}  "
                  f"NDCG={result[f'ndcg@{k}']:.4f}  "
                  f"MAP={result[f'map@{k}']:.4f}  "
                  f"Hit={result[f'hit_rate@{k}']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    df = pd.DataFrame(all_results)
    k = 10
    summary_cols = ['model', f'precision@{k}', f'recall@{k}', f'ndcg@{k}',
                    f'map@{k}', f'hit_rate@{k}', 'coverage', 'time_sec']
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df[available_cols].sort_values(f'ndcg@{k}', ascending=False)

    print(f"\nMetrics at K={k}:")
    print(summary.to_string(index=False))

    output_file = 'evaluation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFull results saved to {output_file}")

    # Markdown
    print(f"\n| Model | Precision@{k} | Recall@{k} | NDCG@{k} | MAP@{k} | Hit Rate@{k} | Coverage |")
    print("|-------|" + "|".join(["--------:"] * 6) + "|")
    for _, row in summary.iterrows():
        print(f"| {row['model']} | "
              f"{row[f'precision@{k}']:.4f} | "
              f"{row[f'recall@{k}']:.4f} | "
              f"{row[f'ndcg@{k}']:.4f} | "
              f"{row[f'map@{k}']:.4f} | "
              f"{row[f'hit_rate@{k}']:.4f} | "
              f"{row['coverage']:.2%} |")


if __name__ == '__main__':
    main()
