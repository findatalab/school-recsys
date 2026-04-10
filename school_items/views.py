import uuid

from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie

from school_items.models import Category, Item

# All 20 real categories from the dataset, translated to Russian.
CATEGORIES_RU = {
    'Binders & Binding Systems': 'Папки и переплётные системы',
    'Book Covers & Book Accessories': 'Обложки для книг',
    'Calendars, Planners & Personal Organizers': 'Календари и ежедневники',
    'Carrying Cases': 'Сумки и чехлы',
    'Cutting & Measuring Devices': 'Режущие и измерительные инструменты',
    'Desk Accessories & Workspace Organizers': 'Настольные аксессуары и органайзеры',
    'Education & Crafts': 'Обучение и рукоделие',
    'Envelopes, Mailers & Shipping Supplies': 'Конверты и почтовые принадлежности',
    'Filing Products': 'Файловые принадлежности',
    'Forms, Recordkeeping & Money Handling': 'Бланки, учёт и кассовые принадлежности',
    'Labels, Indexes & Stamps': 'Этикетки, индексы и штампы',
    'Office & School Supplies': 'Канцелярские и школьные товары',
    'Office Storage Supplies': 'Принадлежности для хранения',
    'Paper': 'Бумага',
    'Presentation Boards': 'Презентационные доски',
    'Staplers & Punches': 'Степлеры и дыроколы',
    'Store Signs & Displays': 'Вывески и стенды',
    'Tape, Adhesives & Fasteners': 'Лента, клей и крепёж',
    'Time Clocks & Cards': 'Табельные часы и карты',
    'Writing & Correction Supplies': 'Письменные и корректирующие принадлежности',
}

ALLOWED_CATEGORIES = set(CATEGORIES_RU.keys())


def _extract_category(categories_en):
    parts = [p.strip() for p in (categories_en or '').split('|') if p.strip()]
    if len(parts) >= 3:
        return parts[2]
    return parts[-1] if parts else ''


def _display_title(item):
    return item.title_ru or item.title_en or item.item_id


def _allowed_items_queryset():
    query = Q()
    for category_name in ALLOWED_CATEGORIES:
        query |= Q(categories_en__icontains=category_name)
    return Item.objects.filter(query).order_by('item_id') if query else Item.objects.all().order_by('item_id')


def _render_catalog(request, category_name=None):
    items = _allowed_items_queryset()
    page_title = 'Школьные товары'

    if category_name:
        items = items.filter(categories_en__icontains=category_name)
        page_title = CATEGORIES_RU.get(category_name, category_name)

    genres = get_genres()
    page_number = request.GET.get("page", 1)
    page, page_end, page_start = handle_pagination(items, page_number)

    context_dict = {
        'movies': page,
        'items': page,
        'items_with_details': _attach_details(page),
        'genres': genres,
        'session_id': session_id(request),
        'user_id': user_id(request),
        'pages': range(page_start, page_end),
        'page_title': page_title,
        'genre_selected': category_name,
    }
    return render(request, 'moviegeek/index.html', context_dict)


@ensure_csrf_cookie
def index(request):
    category_name = request.GET.get('genre') or request.GET.get('category')
    return _render_catalog(request, category_name)


def handle_pagination(items, page_number):
    paginate_by = 60
    paginator = Paginator(items, paginate_by)

    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        page_number = 1
        page = paginator.page(page_number)
    except EmptyPage:
        page_number = paginator.num_pages
        page = paginator.page(page_number)

    page_number = int(page_number)
    page_start = max(1, page_number - 3)
    page_end = min(paginator.num_pages + 1, page_number + 3)
    return page, page_end, page_start


@ensure_csrf_cookie
def category(request, category_id):
    return _render_catalog(request, category_id)


genre = category


def detail(request, item_id):
    genres = get_genres()
    item = Item.objects.filter(item_id=item_id).first()
    category_name = _extract_category(item.categories_en) if item else ''
    category_label = CATEGORIES_RU.get(category_name, category_name) if category_name else ''

    context_dict = {
        'movie_id': item_id,
        'genres': genres,
        'movie_genres': [{'name': category_label}] if category_label else [],
        'movie': item,
        'display_title': _display_title(item) if item else item_id,
        'item': item,
        'session_id': session_id(request),
        'user_id': user_id(request),
    }

    return render(request, 'moviegeek/detail.html', context_dict)


def item_detail_api(request, item_id):
    item = Item.objects.filter(item_id=item_id).first()
    if item is None:
        return JsonResponse({'error': 'Not found'}, status=404)

    category_name = _extract_category(item.categories_en)
    data = {
        'id': item_id,
        'title': _display_title(item),
        'title_en': item.title_en,
        'category': CATEGORIES_RU.get(category_name, category_name),
        'price': item.price,
        'average_rating': item.average_rating,
        'rating_number': item.rating_number,
        'description_en': item.description_en,
    }
    return JsonResponse(data)


def search_for_item(request):
    search_term = request.GET.get('q')

    if not search_term:
        return redirect('/movies/')

    items = Item.objects.filter(
        Q(title_en__icontains=search_term)
        | Q(title_ru__icontains=search_term)
        | Q(description_en__icontains=search_term)
        | Q(categories_en__icontains=search_term)
    ).order_by('item_id')[:60]

    context_dict = {
        'genres': get_genres(),
        'movies': items,
        'items': items,
        'items_with_details': _attach_details(items),
    }

    return render(request, 'moviegeek/search_result.html', context_dict)


search_for_movie = search_for_item


def _attach_details(items):
    result = []
    for item in items:
        result.append({
            'movie_id': item.item_id,
            'title': _display_title(item),
            'price': item.price,
            'average_rating': item.average_rating,
        })
    return result


def get_genres():
    category_names = list(
        Category.objects.filter(name__in=ALLOWED_CATEGORIES)
        .order_by('name')
        .values_list('name', flat=True)
    )

    if not category_names:
        category_names = sorted({
            category_name
            for categories_en in Item.objects.values_list('categories_en', flat=True)
            for category_name in [_extract_category(categories_en)]
            if category_name in ALLOWED_CATEGORIES
        })

    return [
        {
            'name': name,
            'name_ru': CATEGORIES_RU.get(name, name),
        }
        for name in category_names
    ]


def session_id(request):
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid1())
    return request.session["session_id"]


def user_id(request):
    uid = request.GET.get("user_id")

    if uid:
        request.session['user_id'] = uid

    if "user_id" not in request.session:
        request.session['user_id'] = 'AFKZENTNBQ7A7V7UXW5JJI6UGRYQ'

    return request.session['user_id']
