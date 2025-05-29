from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse  # ✅ Ajouté

urlpatterns = [
    path('', lambda request: JsonResponse({'message': 'Bienvenue dans l’API SmartTri 🌍'})),  # ✅ Ajouté
    path('admin/', admin.site.urls),
    path('api/auth/', include('authentication.urls')),
    path('api/scanner/', include('scanner.urls')),
    path('api/ia/', include('ia.urls')),
    path('api/quiz/', include('quiz.urls')),
    path('api/invites/', include('invites.urls')),
]
