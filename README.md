# Рекомендательная система школьных и офисных товаров

Демо-сайт рекомендательных систем на базе Django. Вместо фильмов используется датасет школьных/офисных товаров Amazon (23 000+ товаров, 700 000+ оценок).

На странице каждого товара демонстрируются **9 алгоритмов рекомендаций**:

| # | Алгоритм | Тип | Описание |
|---|----------|-----|----------|
| 1 | Item-based CF | Collaborative Filtering | Похожие товары по матрице сходства (Jaccard) |
| 2 | Neighborhood CF | Collaborative Filtering | Рекомендации на основе похожих пользователей |
| 3 | Content-Based (item) | Content-Based | TF-IDF по описаниям товаров |
| 4 | Content-Based (user) | Content-Based | Профиль интересов пользователя через TF-IDF |
| 5 | ALS | Matrix Factorization | Alternating Least Squares (implicit) |
| 6 | BPR | Matrix Factorization | Bayesian Personalized Ranking (implicit) |
| 7 | SVD | Matrix Factorization | Truncated SVD (scipy) |
| 8 | FWLS | Hybrid | Feature Weighted Linear Stacking (CB + CF) |
| 9 | Popularity | Baseline | Популярные товары в той же категории |

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Создание базы данных и загрузка данных

```bash
python manage.py makemigrations
python manage.py migrate --run-syncdb
```

Датасеты находятся в папке `data/`:

```bash
# Загрузка товаров (23 000+ записей)
python populate_office_school_items.py

# Загрузка оценок пользователей (700 000+ записей)
python populate_office_school_ratings.py
```

### 3. Запуск сервера

```bash
python manage.py runserver 127.0.0.1:8000
```

Сайт доступен по адресу: [http://127.0.0.1:8000](http://127.0.0.1:8000)


### 4. Построение матрицы сходства (Item-based CF)

```bash
python -m builder.item_similarity_calculator
```

### 5. Построение TF-IDF матрицы (Content-Based)

```bash
python build_tfidf_similarity.py
python build_content_similarity.py
```

### 6. Обучение моделей (ALS, BPR, SVD)

```bash
python train_implicit_models.py
```

### 7. Оценка моделей

Скрипт `evaluate_models.py` вычисляет метрики Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K и Coverage для всех моделей:

```bash
python evaluate_models.py --users 100000 --min-ratings 5 --k 5 10
```

Скрипт автоматически переобучает MF-модели (ALS, BPR, SVD) только на train-данных, чтобы исключить утечку данных. Сплит сохраняется в `data/splits/split.json`, результаты — в `data/results/evaluation_results.csv`.

Для повторной оценки на том же сплите:

```bash
python evaluate_models.py --load-split data/splits/split.json --k 5 10
```

## Структура проекта

```
├── data/
│   ├── office_school_items.csv        # Датасет товаров
│   ├── office_school_interactions.csv # Датасет оценок
│   ├── results/                       # Результаты оценки (gitignored)
│   └── splits/                        # Train/test сплиты (gitignored)
├── populate_office_school_items.py    # Скрипт загрузки товаров в БД
├── populate_office_school_ratings.py  # Скрипт загрузки оценок в БД
├── train_implicit_models.py           # Обучение ALS, BPR, SVD
├── evaluate_models.py                 # Оценка всех моделей (с переобучением на train)
├── builder/
│   ├── item_similarity_calculator.py  # Построение Item-based CF
│   └── tfidf_similarity_builder.py    # Построение TF-IDF сходства
├── recs/
│   ├── als_recommender.py             # ALS (implicit)
│   ├── implicit_bpr_recommender.py    # BPR (implicit)
│   ├── svd_recommender.py             # SVD (scipy)
│   ├── neighborhood_based_recommender.py  # Neighborhood CF
│   ├── content_based_recommender.py   # Content-Based (LDA/TF-IDF)
│   ├── fwls_recommender.py            # Hybrid FWLS
│   └── popularity_recommender.py      # Popularity baseline
├── school_items/                      # Django app: модели товаров
├── analytics/                         # Django app: рейтинги
├── recommender/                       # Django app: API рекомендаций
└── templates/                         # HTML шаблоны
```

## Данные

- **Товары**: Amazon Office & School Supplies (title, description, features, categories, price, rating — на EN и RU)
- **Оценки**: user_id, parent_asin, rating (1-5), timestamp
- Категории извлекаются из 3-го уровня поля `categories_en` (split по `|`) и переведены на русский

## Технологии

- Python 3.10+
- Django 4.2+
- implicit (ALS, BPR)
- scipy (SVD)
- scikit-learn (TF-IDF)
- gensim (LDA)
- pandas, numpy
