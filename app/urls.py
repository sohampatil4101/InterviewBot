from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name="home"),
    path('getinterviewtips/',views.InterviewTips.as_view()),
    path('placementprediction/',views.PlacementPrediction.as_view()),
    # path('interviewbot/',views.InterviewChatbot.as_view()),

    
]
