"""
Build content-based similarity using TF-IDF on item descriptions,
features and categories. Populates the LdaSimilarity table.
"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django
django.setup()

import numpy as np
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tqdm import tqdm

from school_items.models import Item
from recommender.models import LdaSimilarity


MIN_SIM = 0.1       # Only save pairs with similarity >= 0.1
TOP_K = 20           # Keep top-K most similar items per item
BATCH_SIZE = 500000  # Bulk create batch size


def build():
    print("Loading item details...")
    items = list(Item.objects.all().values_list(
        'item_id', 'description_en', 'features_en', 'categories_en'
    ))
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items found!")
        return

    item_ids = [r[0] for r in items]

    # Combine description + features + category into one text per item
    texts = []
    for item_id, desc, features, cats in items:
        # Weight categories higher by repeating them
        cat_text = ' '.join(cats.replace('|', ' ').split()) if cats else ''
        feat_text = features.replace('|', ' ') if features else ''
        combined = f"{desc or ''} {feat_text} {cat_text} {cat_text} {cat_text}"
        texts.append(combined)

    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Compute similarities in chunks to avoid memory issues
    n_items = tfidf_matrix.shape[0]
    chunk_size = 1000  # Process 1000 items at a time

    print("Truncating LdaSimilarity table...")
    LdaSimilarity.objects.all().delete()

    today = date.today()
    total_saved = 0
    batch = []

    print(f"Computing cosine similarities (top-{TOP_K} per item, min_sim={MIN_SIM})...")
    for start in tqdm(range(0, n_items, chunk_size)):
        end = min(start + chunk_size, n_items)
        chunk = tfidf_matrix[start:end]

        # Cosine similarity between this chunk and ALL items
        sims = cosine_similarity(chunk, tfidf_matrix)

        for local_idx in range(sims.shape[0]):
            global_idx = start + local_idx
            row = sims[local_idx]

            # Zero out self-similarity
            row[global_idx] = 0

            # Get top-K indices
            if TOP_K < len(row):
                top_indices = np.argpartition(row, -TOP_K)[-TOP_K:]
            else:
                top_indices = np.arange(len(row))

            for j in top_indices:
                sim_val = row[j]
                if sim_val >= MIN_SIM:
                    batch.append(LdaSimilarity(
                        created=today,
                        source=item_ids[global_idx],
                        target=item_ids[j],
                        similarity=round(float(sim_val), 7),
                    ))

                    if len(batch) >= BATCH_SIZE:
                        LdaSimilarity.objects.bulk_create(batch)
                        total_saved += len(batch)
                        batch = []

    if batch:
        LdaSimilarity.objects.bulk_create(batch)
        total_saved += len(batch)

    print(f"Done! Saved {total_saved} content-based similarity pairs.")
    print(f"LdaSimilarity count: {LdaSimilarity.objects.count()}")


if __name__ == '__main__':
    build()
