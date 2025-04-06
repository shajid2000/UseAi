from django.urls import path
from .views import HomeView, UploadKnowledgeBase

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("upload-data-set/", UploadKnowledgeBase.as_view(), name="upload"),
]