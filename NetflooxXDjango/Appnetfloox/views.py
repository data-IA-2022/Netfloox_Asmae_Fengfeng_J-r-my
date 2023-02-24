from django.shortcuts import render
import pandas as pd 
from Appnetfloox.models import Recommendations, Data_Prediction
from .models import Recommendations
from django.shortcuts import render, redirect
from Appnetfloox.recommend import model



if Data_Prediction.objects.count() == 0:
    # Suppression de toutes les entrées de la table
    Data_Prediction.objects.all().delete()
    # Lire le fichier CSV dans un dataframe
    df_pred = pd.read_csv('C:\\Users\\jejel\\Desktop\\Netfloox\\save_de_prediction.csv')
    # Convertir le dataframe en une liste de dictionnaires
    records = df_pred.to_dict('records')
    # Insérer les enregistrements dans la base de données
    Data_Prediction.objects.bulk_create([Data_Prediction(**record) for record in records])

if Recommendations.objects.count() == 0:
    # Suppression de toutes les entrées de la table
    Recommendations.objects.all().delete()
    # Lire le fichier CSV dans un dataframe
    df_reco = pd.read_csv('C:\\Users\\jejel\\Desktop\\Netfloox\\save_de_recommend.csv')
    # Convertir le dataframe en une liste de dictionnaires
    records = df_reco.to_dict('records')
    # Insérer les enregistrements dans la base de données
    Recommendations.objects.bulk_create([Recommendations(**record) for record in records])



def web(request):
    df  = pd.DataFrame(Recommendations.objects.all().values())
    # Récupération des données
    listeFilm=[element[0] for element in Recommendations.objects.values_list('primaryTitle')]
    message = ""
    if request.method == "POST":
        film=request.POST.get('movie')
        if film in listeFilm:
            message = "Le film existe dans la base"
            reco = model(film, df)
            return render(request,"netfloox.html", {"films":listeFilm, "message":message, "reco":reco, "film" : film})
        else:
            message = "Le film n'existe pas dans la base, merci de choisir un film existant"
            return render(request,"netfloox.html", {"films":listeFilm, "message":message})
        
    
    return render(request,"netfloox.html", {"films":listeFilm})


