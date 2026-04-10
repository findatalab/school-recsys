"""prs_project URL Configuration"""
from django.contrib import admin
from django.urls import include, path, re_path

from school_items import views

urlpatterns = [
    path('', views.index, name='index'),
    path('movies/', include('school_items.urls')),
    path('school_items/', include('school_items.urls')),
    path('collect/', include('collector.urls')),
    path('analytics/', include('analytics.urls')),
    re_path(r'^admin/', admin.site.urls),
    path('rec/', include('recommender.urls')),
]
