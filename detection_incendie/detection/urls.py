from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect_fire/', views.detect_fire, name='detect_fire'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
