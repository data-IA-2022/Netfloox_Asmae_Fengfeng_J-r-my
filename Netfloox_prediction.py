# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:28:52 2023

@author: jejel
"""

#---------------------------------------------------------------- Imports DEBUT
import pandas as pd
import pymysql
import numpy as np
import time
import os
import joblib
import tkinter as tk
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

# machine learning - scikit learn:
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
#------------------------------------------------------------------ Imports FIN



#------------------------------------------------------------ BDD Connect DEBUT
try:  
    conn = create_engine('mysql+pymysql://netfloox:987456321@localhost/netfloox')
    print("La connexion à la base de données a été établie avec succès")
except Exception as e:
    print("Impossible de se connecter à la base de données. Raison :", str(e))
#-------------------------------------------------------------- BDD Connect FIN



#-------------------------------------------------------------- BDD Query DEBUT
'''Effacer le fichier save_de_prediction dans le cas ou la requête change'''

if os.path.isfile('save_de_prediction.csv'):
    print('Le fichier existe déjà')
else:
    print('Le fichier n\'existe pas encore')
    df_readed = pd.read_sql_query("""
    SELECT basics.genres, basics.runtimeMinutes, principals.category, name_basics.primaryName, ratings.averageRating, ratings.numVotes 
    FROM basics
    INNER JOIN principals
    ON principals.tconst = basics.tconst
    INNER JOIN ratings
    ON basics.tconst = ratings.tconst 
    INNER JOIN netfloox.name_basics
    ON principals.nconst = name_basics.nconst1
    WHERE basics.titleType ='movie';
    """ , conn)
    
if not os.path.isfile('save_de_prediction.csv'):
    df_readed.to_csv('save_de_prediction.csv', index=False)
    print('Le fichier CSV a été sauvegardé avec succès')

df_pour_prediction=pd.read_csv("save_de_prediction.csv")
    
print(f"\n{df_pour_prediction.columns}")
print(f"\n{df_pour_prediction.dtypes}\n")
print(f"{df_pour_prediction.shape}\n")
#---------------------------------------------------------------- BDD Query FIN



#---------------------------------------------------------- BDD Formatage DEBUT
df_pour_prediction = df_pour_prediction.replace('\\N', np.nan)
df_pour_prediction = df_pour_prediction.dropna()
df_pour_prediction = df_pour_prediction.astype({'genres': str, 'runtimeMinutes': float, 'category': str, 'primaryName': str})
print(f"\n{df_pour_prediction.dtypes}\n")
# moyenne = df_pour_prediction['runtimeMinutes'].mean()
# print(moyenne)
#------------------------------------------------------------ BDD Formatage FIN



#------------------------------------------------- Entrainement du modèle DEBUT

#Définition de notre X(données) et y(target) pour la prédiction du rating
y = df_pour_prediction['averageRating']
X = df_pour_prediction.drop(['averageRating'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
print("La longueur du dataset de base :", len(X))
print("La longueur du dataset d'entraînement :", len(X_train))
print("La longueur du dataset de test :", len(X_test))


preparation = ColumnTransformer(
    transformers=[       
        ('data_cat', OneHotEncoder(handle_unknown='ignore'), ['category', 'primaryName']), 
        ('data_tex', CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', ')), 'genres'),
        ('data_num', StandardScaler(), ['runtimeMinutes', 'numVotes'])
    ])

X_train_prepared = preparation.fit_transform(X_train)
X_test_prepared = preparation.transform(X_test)

print("\n-----------------------------Transformers-----------------------------\n")
print(f"{preparation}")
print("\n-----------------------------Transformers-----------------------------")


# Créer un dictionnaire de modèles et de paramètres
models = {
    'LinearRegression': LinearRegression(),
    # 'LogisticRegression': LogisticRegression(),
    # 'RandomForestRegressor': RandomForestRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
}

params = {
    'LinearRegression': {'normalize': [True, False]},
    # 'LogisticRegression': {'C': [0.1, 1, 10]},
    # 'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10]},
    'KNeighborsRegressor': {'n_neighbors': [3, 5, 7]},
}


if os.path.isfile('best_model.joblib'):
     os.remove('best_model.joblib')

# Créer un objet GridSearchCV avec le dictionnaire de modèles et de paramètres
p=0
for model_name, model in models.items():
    p=p+1
    print(f"\n-----------------------------Modèle {p}---------------------------------\n")
    print(f"GridSearchCV for {model_name}")
    start_time = time.time()
    gs = GridSearchCV(model, params[model_name], cv=5)
    gs.fit(X_train_prepared, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Best params: {gs.best_params_}")
    print(f"Train score: {gs.best_score_*100:.3f}%")
    print(f"Test score: {gs.score(X_test_prepared, y_test)*100:.3f}%\n")    
    print(f"\n------------------------------------------------------- {training_time:.2f} seconds")

    best_model = gs.best_estimator_
    joblib.dump(best_model, 'best_model.joblib')    
#--------------------------------------------------- Entrainement du modèle FIN


#------------------------------------------------------------- Prédiction DEBUT


# Créer une fenêtre
window = tk.Tk()

# Définir la taille de la fenêtre
window.geometry('400x600')

# Verrouiller la taille de la fenêtre
window.resizable(False, False)

# Forcer l'affichage de la fenêtre au premier plan
window.lift()


# Créer des listes contenant toutes les possibilités existantes des colonnes du jeu de données.
all_genres = df_pour_prediction['genres'].unique().tolist()
all_genres.sort()
all_names = df_pour_prediction['primaryName'].unique().tolist()
all_names.sort()
all_category = df_pour_prediction['category'].unique().tolist()
all_category.sort()

# Créer une variable pour stocker la sélection de l'utilisateur.
selected_genre = tk.StringVar(window)
selected_name = tk.StringVar(window)
selected_category = tk.StringVar(window)

# Ajouter un champ de saisie pour l'utilisateur
# Ajouter une liste déroulante à la fenêtre, en utilisant la liste créée à l'étape 1 pour définir les options de la liste.

genres_label = tk.Label(window, text='Choisissez un genre :')
genres_label.pack()
genre_dropdown = tk.OptionMenu(window, selected_genre, *all_genres)
genre_dropdown.place(x=100, y=0)
genre_dropdown.pack()

nom_label = tk.Label(window, text='Choisissez un acteur/réalisateur/écrivain :')
nom_label.pack()
nom_dropdown = tk.OptionMenu(window, selected_name, *all_names)
nom_dropdown.place(x=100, y=0)
nom_dropdown.pack()

category_label = tk.Label(window, text='Définir sa catégorie :')
category_label.pack()
category_dropdown = tk.OptionMenu(window, selected_category, *all_category)
category_dropdown.place(x=100, y=0)
category_dropdown.pack()

duree_label = tk.Label(window, text='Entrez la durée du film :')
duree_label.pack()
duree_window = tk.Entry(window)
duree_window.pack()

votes_label = tk.Label(window, text='Entrez le nombre de votes :')
votes_label.pack()
votes_window = tk.Entry(window)
votes_window.pack()


# Ajouter un bouton pour valider la saisie
def validate_input():
    print("\n---------------------Configuration de la prédiction------------------------\n")
    genres = selected_genre.get()  # Récupérer la valeur sélectionnée par l'utilisateur
    print(f"Le genre sélectionné est : {genres}")
    
    name = selected_name.get()  # Récupérer la valeur sélectionnée par l'utilisateur
    print(f"Le nom sélectionné est : {name}")
    
    category = selected_category.get()  # Récupérer la valeur sélectionnée par l'utilisateur
    print(f"La categorie sélectionné est : {name}")
    
    duree = duree_window.get()  # Récupérer la valeur sélectionnée par l'utilisateur
    print(f"La durée du film est : {duree}")
    
    votes = votes_window.get()  # Récupérer la valeur sélectionnée par l'utilisateur
    print(f"Le nombre de votes estimé est : {votes}")
    print("\n---------------------------------------------------------------------------\n")

    
    # Charger le meilleur modèle à partir du fichier
    best_model = joblib.load('best_model.joblib')
    
    # Entrer des éléments d'entrée pour la prédiction
    # input_data = {
    #     'category': 'actor',
    #     'primaryName': 'Tom Cruise',
    #     'genres': 'Romance',
    #     'runtimeMinutes': 120,
    #     'numVotes': 10000
    # }
    
    input_data = {
        'category': category,
        'primaryName': name,
        'genres': genres,
        'runtimeMinutes': duree,
        'numVotes': votes
    }
    
    # Convertir les éléments d'entrée en dataframe
    input_df = pd.DataFrame([input_data])
    
    # Préparer les données d'entrée
    input_prepared = preparation.transform(input_df)
    
    # Faire la prédiction
    prediction = best_model.predict(input_prepared)
    
    for widget in frame.winfo_children():
        widget.destroy()
    print(f"La prédiction de rating est : {prediction[0]:.2f}")
    label = tk.Label(frame, text=f"La prédiction de rating est : {prediction[0]:.2f}")
    label.pack()
    
button = tk.Button(window, text='Valider', command=validate_input)
button.pack()

# Ajouter un label pour afficher le résultat de la saisie
result_label = tk.Label(window, text='')
result_label.pack()

# Ajouter un frame pour y ajouter les labels des propositions
frame = tk.Frame(window)
frame.pack()

# Fonction pour quitter la fenêtre et arrêter le programme
def quit_program():
    window.quit()
    window.destroy()

# Afficher la fenêtre
window.mainloop()

#-------------------------------------------------------------- Prédiction FIN



















