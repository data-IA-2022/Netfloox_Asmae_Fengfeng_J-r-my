from django import forms
from .models import Recommendations


class RecommendationForm(forms.ModelForm):
    class Meta:
        model = Recommendations
        fields = ['primaryTitle']