from recs.base_recommender import base_recommender
from analytics.models import Rating
from recommender.models import Similarity
from django.db.models import Q

from decimal import Decimal
from collections import defaultdict


class NeighborhoodBasedRecs(base_recommender):

    def __init__(self, neighborhood_size=15, min_sim=0.0):
        self.neighborhood_size = neighborhood_size
        self.min_sim = min_sim
        self.max_candidates = 200

    def recommend_items(self, user_id, num=6):
        active_user_items = Rating.objects.filter(user_id=user_id).order_by('-rating')[:100]
        return self.recommend_items_by_ratings(user_id, active_user_items.values(), num)

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):
        if len(active_user_items) == 0:
            return []

        movie_ids = {movie['movie_id']: Decimal(str(movie['rating'])) for movie in active_user_items}
        user_mean = sum(movie_ids.values()) / len(movie_ids)

        candidate_items = list(
            Similarity.objects.filter(
                Q(source__in=movie_ids.keys())
                & ~Q(target__in=movie_ids.keys())
                & Q(similarity__gt=self.min_sim)
            ).order_by('-similarity')[:self.max_candidates]
        )

        # Group candidates by target item
        target_sims = defaultdict(list)
        for c in candidate_items:
            target_sims[c.target].append(c)

        recs = {}
        for target, sims in target_sims.items():
            # Use top-N most similar neighbors
            top_sims = sims[:self.neighborhood_size]

            pre = Decimal(0)
            sim_sum = Decimal(0)
            for sim_item in top_sims:
                r = movie_ids[sim_item.source] - user_mean
                pre += sim_item.similarity * r
                sim_sum += sim_item.similarity

            if sim_sum > 0:
                recs[target] = {'prediction': user_mean + pre / sim_sum,
                                'sim_items': [s.source for s in top_sims]}

        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
        return sorted_items

    def predict_score(self, user_id, item_id):
        user_items = Rating.objects.filter(user_id=user_id)
        user_items = user_items.exclude(movie_id=item_id).order_by('-rating')[:100]
        movie_ids = {movie.movie_id: movie.rating for movie in user_items}
        return self.predict_score_by_ratings(item_id, movie_ids)

    def predict_score_by_ratings(self, item_id, movie_ids):
        top = Decimal(0.0)
        bottom = Decimal(0.0)
        ids = movie_ids.keys()
        mc = self.max_candidates
        candidate_items = (Similarity.objects.filter(source__in=ids)
                                             .exclude(source=item_id)
                                             .filter(target=item_id))
        candidate_items = candidate_items.distinct().order_by('-similarity')[:mc]

        if len(candidate_items) == 0:
            return 0

        for sim_item in candidate_items:
            r = movie_ids[sim_item.source]
            top += sim_item.similarity * r
            bottom += sim_item.similarity

        return Decimal(top / bottom)
