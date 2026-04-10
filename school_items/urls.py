from django.urls import path, re_path

from school_items import views

urlpatterns = [
    path('', views.index, name='index'),
    path('item/<str:item_id>/', views.detail, name='detail'),
    path('movie/<str:item_id>/', views.detail, name='detail_legacy'),
    re_path(r'^category/(?P<category_id>[\w\s,&-]+)/$', views.category, name='category'),
    re_path(r'^genre/(?P<category_id>[\w\s,&-]+)/$', views.category, name='genre'),
    path('search/', views.search_for_item, name='search_for_item'),
    path('api/item/<str:item_id>/', views.item_detail_api, name='item_detail_api'),
]
