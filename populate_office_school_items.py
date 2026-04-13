import csv
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django

django.setup()

from tqdm import tqdm

from school_items.models import Category, Item


def get_category(categories_en):
    """Extract the primary catalog category from a pipe-separated path."""
    parts = [p.strip() for p in (categories_en or '').split('|') if p.strip()]
    if len(parts) >= 3:
        return parts[2]
    return parts[-1] if parts else 'Other'


def delete_db():
    print("Truncating existing data...")
    Item.objects.all().delete()
    Category.objects.all().delete()
    print("Done truncating.")


def populate():
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'office_school_items.csv',
    )

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} items to load.")

    for row in tqdm(rows):
        asin = row['parent_asin']
        categories_en = row.get('categories_en', '')
        category_name = get_category(categories_en)
        Category.objects.get_or_create(name=category_name)

        try:
            price = float(row['price']) if row.get('price') else None
        except (ValueError, TypeError):
            price = None

        try:
            avg_rating = float(row['average_rating']) if row.get('average_rating') else None
        except (ValueError, TypeError):
            avg_rating = None

        try:
            rating_number = int(row['rating_number']) if row.get('rating_number') else 0
        except (ValueError, TypeError):
            rating_number = 0

        Item.objects.update_or_create(
            item_id=asin,
            defaults={
                'title_en': row.get('title_en', ''),
                'title_ru': row.get('title_ru', ''),
                'description_en': row.get('description_en', ''),
                'features_en': row.get('features_en', ''),
                'categories_en': categories_en,
                'description_ru': row.get('description_ru', ''),
                'features_ru': row.get('features_ru', ''),
                'categories_ru': row.get('categories_ru', ''),
                'price': price,
                'average_rating': avg_rating,
                'rating_number': rating_number,
            },
        )

    print(f"Loaded {Item.objects.count()} items, {Category.objects.count()} categories.")


if __name__ == '__main__':
    print("Starting Office & School Items population...")
    delete_db()
    populate()
    print("Done.")
