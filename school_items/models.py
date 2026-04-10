from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=128, unique=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Item(models.Model):
    item_id = models.CharField(max_length=32, unique=True, primary_key=True)
    title_en = models.CharField(max_length=512, blank=True, default='')
    title_ru = models.CharField(max_length=512, blank=True, default='')
    description_en = models.TextField(blank=True)
    features_en = models.TextField(blank=True)
    categories_en = models.CharField(max_length=512, blank=True)
    description_ru = models.TextField(blank=True)
    features_ru = models.TextField(blank=True)
    categories_ru = models.CharField(max_length=512, blank=True)
    price = models.FloatField(null=True, blank=True)
    average_rating = models.FloatField(null=True, blank=True)
    rating_number = models.IntegerField(default=0)

    class Meta:
        db_table = 'item_detail'

    @property
    def title(self):
        return self.title_ru or self.title_en or self.item_id

    @property
    def movie_id(self):
        return self.item_id

    @property
    def primary_category(self):
        parts = [p.strip() for p in (self.categories_en or '').split('|') if p.strip()]
        if len(parts) >= 3:
            return parts[2]
        return parts[-1] if parts else ''

    def __str__(self):
        return self.title
