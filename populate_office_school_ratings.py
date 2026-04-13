import os
import csv
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django
django.setup()

from tqdm import tqdm
from analytics.models import Rating


def delete_db():
    print("Truncating ratings...")
    Rating.objects.all().delete()
    print("Done truncating.")


def populate():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'office_school_interactions.csv')

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} ratings to load.")

    batch = []
    for row in tqdm(rows):
        ts_ms = int(row['timestamp'])
        ts = datetime.fromtimestamp(ts_ms / 1000)

        batch.append(Rating(
            user_id=row['user_id'],
            movie_id=row['parent_asin'],
            rating=float(row['rating']),
            rating_timestamp=ts,
            type='explicit',
        ))

        if len(batch) >= 1000:
            Rating.objects.bulk_create(batch)
            batch = []

    if batch:
        Rating.objects.bulk_create(batch)

    print(f"Loaded {Rating.objects.count()} ratings.")


if __name__ == '__main__':
    print("Starting Office & School Ratings population...")
    delete_db()
    populate()
    print("Done.")
