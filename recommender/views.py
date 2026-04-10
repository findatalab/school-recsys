import operator
import logging
from decimal import Decimal
from math import sqrt

import numpy as np
from django.db.models import Avg, Count, Q
from django.http import JsonResponse

from analytics.models import Rating
from collector.models import Log
from recommender.models import SeededRecs, Similarity
from school_items.models import Item
from recs.neighborhood_based_recommender import NeighborhoodBasedRecs
from recs.popularity_recommender import PopularityBasedRecs

logger = logging.getLogger(__name__)


def _display_title(item):
    return item.title_ru or item.title_en or item.item_id


def _same_category_items(item, exclude_ids=None):
    if not item:
        return Item.objects.none()

    category_name = item.primary_category
    if not category_name:
        return Item.objects.none()

    qs = Item.objects.filter(categories_en__icontains=category_name)
    if exclude_ids:
        qs = qs.exclude(item_id__in=exclude_ids)
    return qs


def _normalize_tuple_recs(sorted_items):
    """Convert [(item_id, {prediction, ...}), ...] to [{target: item_id}, ...]"""
    if not sorted_items:
        return []
    if isinstance(sorted_items, dict):
        return [{'target': k} for k in sorted_items.keys()]
    return [{'target': item[0]} for item in sorted_items]


def get_association_rules_for(request, content_id, take=6):
    data = SeededRecs.objects.filter(source=content_id) \
               .order_by('-confidence') \
               .values('target', 'confidence', 'support')[:take]

    return JsonResponse(dict(data=list(data)), safe=False)


def recs_using_association_rules(request, user_id, take=6):
    events = Log.objects.filter(user_id=user_id)\
                        .order_by('created')\
                        .values_list('content_id', flat=True)\
                        .distinct()

    seeds = set(events[:20])

    rules = SeededRecs.objects.filter(source__in=seeds) \
        .exclude(target__in=seeds) \
        .values('target') \
        .annotate(confidence=Avg('confidence')) \
        .order_by('-confidence')

    recs = [{'id': rule['target'],
             'confidence': rule['confidence']} for rule in rules]

    return JsonResponse(dict(data=list(recs[:take])))


def chart(request, take=10):
    sorted_items = PopularityBasedRecs().recommend_items_from_log(take)
    ids = [i['content_id'] for i in sorted_items]

    titles = {
        item.item_id: _display_title(item)
        for item in Item.objects.filter(item_id__in=ids)
    }

    sorted_items = [
        {
            'movie_id': item['content_id'],
            'title': titles.get(item['content_id'], item['content_id']),
        }
        for item in sorted_items
    ]
    data = {
        'data': sorted_items
    }

    return JsonResponse(data, safe=False)


def pearson(users, this_user, that_user):
    if this_user in users and that_user in users:
        this_user_avg = sum(users[this_user].values()) / len(users[this_user].values())
        that_user_avg = sum(users[that_user].values()) / len(users[that_user].values())

        all_movies = set(users[this_user].keys()) & set(users[that_user].keys())

        dividend = 0
        a_divisor = 0
        b_divisor = 0
        for movie in all_movies:

            if movie in users[this_user].keys() and movie in users[that_user].keys():
                a_nr = users[this_user][movie] - this_user_avg
                b_nr = users[that_user][movie] - that_user_avg
                dividend += a_nr * b_nr
                a_divisor += pow(a_nr, 2)
                b_divisor += pow(b_nr, 2)

        divisor = Decimal(sqrt(a_divisor) * sqrt(b_divisor))
        if divisor != 0:
            return dividend / Decimal(sqrt(a_divisor) * sqrt(b_divisor))

    return 0


def jaccard(users, this_user, that_user):
    if this_user in users and that_user in users:
        intersect = set(users[this_user].keys()) & set(users[that_user].keys())

        union = set(users[this_user].keys()) | set(users[that_user].keys())

        return len(intersect) / Decimal(len(union))
    else:
        return 0


def similar_users(request, user_id, sim_method):
    min = request.GET.get('min', 1)

    ratings = Rating.objects.filter(user_id=user_id)
    sim_users = Rating.objects.filter(movie_id__in=ratings.values('movie_id')) \
        .values('user_id') \
        .annotate(intersect=Count('user_id')).filter(intersect__gt=min)

    dataset = Rating.objects.filter(user_id__in=sim_users.values('user_id'))

    users = {u['user_id']: {} for u in sim_users}

    for row in dataset:
        if row.user_id in users.keys():
            users[row.user_id][row.movie_id] = row.rating

    similarity = dict()

    switcher = {
        'jaccard': jaccard,
        'pearson': pearson,
    }

    for user in sim_users:
        func = switcher.get(sim_method, lambda: "nothing")
        s = func(users, user_id, user['user_id'])

        if s > 0.2:
            similarity[user['user_id']] = round(s, 2)
    topn = sorted(similarity.items(), key=operator.itemgetter(1), reverse=True)[:10]

    data = {
        'user_id': user_id,
        'num_movies_rated': len(ratings),
        'type': sim_method,
        'topn': topn,
        'similarity': topn,
    }

    return JsonResponse(data, safe=False)


def similar_content(request, content_id, num=6):
    try:
        from recs.content_based_recommender import ContentBasedRecs
        sorted_items = ContentBasedRecs().seeded_rec([content_id], num)
        if sorted_items:
            return JsonResponse({'source_id': content_id, 'data': sorted_items}, safe=False)
    except Exception as e:
        logger.warning(f"Content-based seeded rec failed: {e}")

    # Fallback: category-based content similarity
    data = _category_based_recs(content_id, num)
    return JsonResponse({'data': data}, safe=False)


def recs_cb(request, user_id, num=6):
    try:
        from recs.content_based_recommender import ContentBasedRecs
        sorted_items = ContentBasedRecs().recommend_items(user_id, num)
        if sorted_items:
            return JsonResponse({'user_id': user_id, 'data': _normalize_tuple_recs(sorted_items)}, safe=False)
    except Exception as e:
        logger.warning(f"Content-based user rec failed: {e}")

    # Fallback: recommend from categories of items the user rated highly
    data = _cb_user_fallback(user_id, num)
    return JsonResponse({'user_id': user_id, 'data': data}, safe=False)


def recs_fwls(request, user_id, num=6):
    try:
        from recs.fwls_recommender import FeatureWeightedLinearStacking
        sorted_items = FeatureWeightedLinearStacking().recommend_items(user_id, num)
        data = {
            'user_id': user_id,
            'data': _normalize_tuple_recs(sorted_items)
        }
    except Exception as e:
        logger.warning(f"FWLS rec failed: {e}")
        data = {'user_id': user_id, 'data': []}

    return JsonResponse(data, safe=False)


def recs_als(request, user_id, num=6):
    try:
        from recs.als_recommender import ALSRecs
        sorted_items = ALSRecs().recommend_items(user_id, num)
        data = {
            'user_id': user_id,
            'data': _normalize_tuple_recs(sorted_items)
        }
    except Exception as e:
        logger.warning(f"ALS rec failed: {e}")
        data = {'user_id': user_id, 'data': []}

    return JsonResponse(data, safe=False)


def recs_bpr(request, user_id, num=6):
    try:
        from recs.implicit_bpr_recommender import ImplicitBPRRecs
        sorted_items = ImplicitBPRRecs().recommend_items(user_id, num)
        data = {
            'user_id': user_id,
            'data': _normalize_tuple_recs(sorted_items)
        }
    except Exception as e:
        logger.warning(f"BPR rec failed: {e}")
        data = {'user_id': user_id, 'data': []}

    return JsonResponse(data, safe=False)


def recs_svd(request, user_id, num=6):
    try:
        from recs.svd_recommender import SVDRecs
        sorted_items = SVDRecs().recommend_items(user_id, num)
        data = {
            'user_id': user_id,
            'data': _normalize_tuple_recs(sorted_items)
        }
    except Exception as e:
        logger.warning(f"SVD rec failed: {e}")
        data = {'user_id': user_id, 'data': []}

    return JsonResponse(data, safe=False)


def recs_cf(request, user_id, num=6):
    try:
        min_sim = float(request.GET.get('min_sim', 0.0))
        sorted_items = NeighborhoodBasedRecs(min_sim=min_sim).recommend_items(user_id, num)
        data = {
            'user_id': user_id,
            'data': _normalize_tuple_recs(sorted_items)
        }
    except Exception as e:
        logger.warning(f"CF rec failed: {e}")
        data = {'user_id': user_id, 'data': []}

    return JsonResponse(data, safe=False)


def recs_pop(request, content_id, num=6):
    """Popularity-based recs from the same category as the given item."""
    item = Item.objects.filter(item_id=content_id).first()
    if not item:
        return JsonResponse({'data': []}, safe=False)

    same_cat_ids = _same_category_items(item, exclude_ids=[content_id]).values_list('item_id', flat=True)

    pop_items = Rating.objects.filter(movie_id__in=same_cat_ids) \
        .values('movie_id') \
        .annotate(cnt=Count('user_id'), avg=Avg('rating')) \
        .order_by('-cnt')[:num]

    data = [{'target': item['movie_id']} for item in pop_items]
    return JsonResponse({'data': data}, safe=False)


def recs_item_similarity(request, content_id, num=6):
    """Item-based CF: return most similar items from the Similarity table."""
    sims = Similarity.objects.filter(source=content_id) \
               .order_by('-similarity') \
               .values('target', 'similarity')[:num]

    return JsonResponse({'data': list(sims)}, safe=False)


def _category_based_recs(content_id, num=6):
    """Content-based fallback: items from the same category, sorted by rating."""
    item = Item.objects.filter(item_id=content_id).first()
    if not item:
        return []

    top_items = _same_category_items(item, exclude_ids=[content_id]) \
        .exclude(average_rating__isnull=True) \
        .order_by('-average_rating', '-rating_number')[:num]

    return [{'target': item.item_id} for item in top_items]


def _cb_user_fallback(user_id, num=6):
    """Content-based user fallback: top-rated items from categories the user likes."""
    user_ratings = Rating.objects.filter(user_id=user_id).order_by('-rating')[:20]
    rated_ids = [r.movie_id for r in user_ratings]
    if not rated_ids:
        return []

    category_counts = {}
    for item in Item.objects.filter(item_id__in=rated_ids):
        category_name = item.primary_category
        if category_name:
            category_counts[category_name] = category_counts.get(category_name, 0) + 1

    top_categories = [
        category_name
        for category_name, _ in sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    if not top_categories:
        return []

    category_query = Q()
    for category_name in top_categories:
        category_query |= Q(categories_en__icontains=category_name)

    cat_item_ids = list(
        Item.objects.filter(category_query)
        .exclude(item_id__in=rated_ids)
        .values_list('item_id', flat=True)
        .distinct()[:500]
    )

    top_items = Item.objects.filter(item_id__in=cat_item_ids) \
        .exclude(average_rating__isnull=True) \
        .order_by('-average_rating', '-rating_number')[:num]

    return [{'target': item.item_id} for item in top_items]


def lda2array(lda_vector, len):
    vec = np.zeros(len)
    for coor in lda_vector:
        if coor[0] > 1270:
            print("auc")
        vec[coor[0]] = coor[1]

    return vec
