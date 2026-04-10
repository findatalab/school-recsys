"""
Build TF-IDF content similarity and store in LdaSimilarity table.
Uses description_en + features_en + categories_en for each item.
"""
import os
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")
import django
django.setup()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from school_items.models import Item
from recommender.models import LdaSimilarity


def build():
    print("Loading items...")
    items = list(Item.objects.all().values('item_id', 'description_en', 'features_en', 'categories_en'))
    print(f"Loaded {len(items)} items")

    # Build text corpus: combine description + features + categories
    item_ids = []
    corpus = []
    for item in items:
        text_parts = []
        if item['description_en']:
            text_parts.append(item['description_en'])
        if item['features_en']:
            text_parts.append(item['features_en'])
        if item['categories_en']:
            text_parts.append(item['categories_en'])
        text = ' '.join(text_parts).strip()
        if text:
            item_ids.append(item['item_id'])
            corpus.append(text)

    print(f"Items with text: {len(corpus)}")

    print("Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = tfidf.fit_transform(corpus)
    print(f"TF-IDF shape: {tfidf_matrix.shape}")

    # Compute similarities in batches (full matrix would be too large)
    TOP_N = 20  # store top-N similar items per item
    BATCH_SIZE = 500
    today = datetime.date.today()

    # Clear old data
    print("Clearing old LdaSimilarity data...")
    LdaSimilarity.objects.all().delete()

    total_pairs = 0
    n_items = len(item_ids)

    for start in range(0, n_items, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_items)
        batch_matrix = tfidf_matrix[start:end]

        # Compute cosine similarity of this batch against all items
        sims = cosine_similarity(batch_matrix, tfidf_matrix)

        records = []
        for i, row_idx in enumerate(range(start, end)):
            source_id = item_ids[row_idx]
            # Get top-N (excluding self)
            sim_scores = sims[i]
            sim_scores[row_idx] = -1  # exclude self

            top_indices = np.argsort(sim_scores)[-TOP_N:][::-1]

            for idx in top_indices:
                score = sim_scores[idx]
                if score > 0.05:  # minimum threshold
                    records.append(LdaSimilarity(
                        created=today,
                        source=source_id,
                        target=item_ids[idx],
                        similarity=round(float(score), 7),
                    ))

        if records:
            LdaSimilarity.objects.bulk_create(records, batch_size=5000)
            total_pairs += len(records)

        print(f"  Batch {start}-{end}: {len(records)} pairs (total: {total_pairs})")

    print(f"\nDone! Total LdaSimilarity records: {total_pairs}")
    print(f"DB count: {LdaSimilarity.objects.count()}")


if __name__ == '__main__':
    build()
