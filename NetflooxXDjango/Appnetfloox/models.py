from django.db import models

# Create your models here.
class Data_Prediction(models.Model):
    genres = models.CharField(max_length=100)
    runtimeMinutes = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    primaryName = models.CharField(max_length=100)
    averageRating = models.FloatField()
    numVotes = models.IntegerField()

class Recommendations(models.Model):
    id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=100)
    primaryName = models.CharField(max_length=100)
    primaryTitle = models.CharField(max_length=100)
    genres = models.CharField(max_length=100)
    startYear = models.FloatField()
