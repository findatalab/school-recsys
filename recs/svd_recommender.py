"""SVD recommender using scipy truncated SVD with bias decomposition."""
import os
import pickle
from decimal import Decimal

import numpy as np

from analytics.models import Rating
from recs.base_recommender import base_recommender

MODEL_DIR = './models'


class SVDRecs(base_recommender):

    def __init__(self):
        self.U_sigma = None
        self.Vt = None
        self.global_mean = 0.0
        self.user_bias = None
        self.item_bias = None
        self.user_to_idx = None
        self.item_map = None
        self.item_to_idx = None
        self._load()

    def _load(self):
        try:
            svd_dir = os.path.join(MODEL_DIR, 'svd')
            U = np.load(os.path.join(svd_dir, 'U.npy'))
            sigma = np.load(os.path.join(svd_dir, 'sigma.npy'))
            self.Vt = np.load(os.path.join(svd_dir, 'Vt.npy'))
            self.global_mean = float(np.load(os.path.join(svd_dir, 'global_mean.npy'))[0])
            self.user_bias = np.load(os.path.join(svd_dir, 'user_bias.npy'))
            self.item_bias = np.load(os.path.join(svd_dir, 'item_bias.npy'))
            self.U_sigma = U * sigma[np.newaxis, :]

            with open(os.path.join(MODEL_DIR, 'user_to_idx.pkl'), 'rb') as f:
                self.user_to_idx = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'item_map.pkl'), 'rb') as f:
                self.item_map = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'item_to_idx.pkl'), 'rb') as f:
                self.item_to_idx = pickle.load(f)
        except FileNotFoundError:
            self.U_sigma = None

    def recommend_items(self, user_id, num=6):
        if self.U_sigma is None or user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        latent = self.U_sigma[user_idx] @ self.Vt
        scores = self.global_mean + self.user_bias[user_idx] + self.item_bias + latent

        rated = set(
            Rating.objects.filter(user_id=user_id)
            .values_list('movie_id', flat=True)
        )

        top_indices = np.argsort(-scores)
        result = []
        for idx in top_indices:
            item_id = self.item_map.get(idx)
            if item_id and item_id not in rated:
                result.append((item_id, {'prediction': Decimal(float(scores[idx]))}))
            if len(result) >= num:
                break
        return result

    def predict_score(self, user_id, item_id):
        if self.U_sigma is None:
            return Decimal(0)
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return Decimal(0)
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        latent = float(self.U_sigma[user_idx] @ self.Vt[:, item_idx])
        score = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + latent
        return Decimal(float(score))
