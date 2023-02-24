from django.contrib import admin
from Appnetfloox.models import Recommendations, Data_Prediction

class colonneRecommendation (admin.ModelAdmin):
    list_display = [champ.name for champ in Recommendations._meta.get_fields()]

admin.site.register(Data_Prediction)
admin.site.register(Recommendations, colonneRecommendation)
