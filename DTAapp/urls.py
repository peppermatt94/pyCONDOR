
from django.urls import path
from pathlib import Path
from . import views

urlpatterns = [
            path('dta', views.DTA_webAPP, name='DTA'),
            path('documentation', views.documentation, name = "Doc DTA"),
            path("source_code", views.source_code, name ="source_code"),
            path("download_dta", views.download_result, name = ""),   
            path("download_fit", views.download_fit, name = ""),
            path("<str:filename>", views.functionpy, name="source_code_txt")
            ]
