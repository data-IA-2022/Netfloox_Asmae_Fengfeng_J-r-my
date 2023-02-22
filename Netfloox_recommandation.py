# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:54:13 2023

@author: jejel
"""

#---------------------------------------------------------------- Imports DEBUT
import pandas as pd
import pymysql
import numpy as np
import tkinter as tk
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
#------------------------------------------------------------------ Imports FIN



#------------------------------------------------------------ BDD Connect DEBUT
try:  
    conn = create_engine('mysql+pymysql://netfloox:987456321@localhost/netfloox')
    print("La connexion à la base de données a été établie avec succès")
except Exception as e:
    print("Impossible de se connecter à la base de données. Raison :", str(e))
#-------------------------------------------------------------- BDD Connect FIN


#------------------------------------------------------------------ pivot table
df_recommend=pd.read_sql_query("""
    SELECT principals_joined.category, principals_joined.primaryName, basics.primaryTitle, basics.genres, basics.startYear  
    FROM principals_joined
    JOIN basics on principals_joined.tconst=basics.tconst 
    WHERE basics.titleType = 'movie'
    ORDER by primaryTitle""" , conn)
 
global df
grouped = df_recommend.groupby(['primaryTitle', 'category'])['primaryName'].apply(lambda x: ', '.join(x[:3]))
data_pivoted = grouped.unstack()
df_recommend.drop(['category', 'primaryName'],axis=1, inplace=True)
df=pd.merge(df_recommend, data_pivoted, on='primaryTitle')

#-------------------------------------------------------------- pivot table FIN


#------------------------------------------------------------------- Main DEBUT

def concat_features(row):
  return(str(row['genres']).replace(",", " ") + " " + str(row['startYear']).replace(" ", "") + " " + str(row['actor']).replace(" ", "") + " " +
          str(row["actress"]).replace(" ", "") + " " + str(row["cinematographer"]).replace(" ", "") + " " + str(row["composer"]).replace(" ", "") + " " +
          str(row["director"]).replace(" ", "") + " " + str(row["editor"]).replace(" ", "") + " " + str(row["producer"]).replace(" ", "") + " " +
          str(row["production_designer"]).replace(" ", "") + " " + str(row["self"]).replace(" ", "") + " " + str(row["writer"]).replace(" ", "") + " " +
          str(row["archive_footage"]).replace(" ", ""))

df = df.replace('\\N', 'nan')
# df = df.fillna('')
df["movie_features"] = df.apply(concat_features, axis=1)
df = df.drop_duplicates(subset=['movie_features'])
print(df.head(50))

# Créer une fenêtre
window = tk.Tk()

# Définir la taille de la fenêtre
window.geometry('400x300')

# Verrouiller la taille de la fenêtre
window.resizable(False, False)

# Forcer l'affichage de la fenêtre au premier plan
window.lift()

# Ajouter un label pour demander à l'utilisateur de saisir un texte
label = tk.Label(window, text='Entrez un texte :')
label.pack()

# Ajouter un champ de saisie pour l'utilisateur
entry = tk.Entry(window)
entry.pack()

# Ajouter un bouton pour valider la saisie
def validate_input():
    global text
    text = entry.get()  # Récupérer la valeur saisie par l'utilisateur
    
    x=0
    
    for i in df['primaryTitle']:
        if text==i:
            movie_index=x
            break
        else:
            x=x+1
        
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["movie_features"])
    sparse_count_matrix = csr_matrix(count_matrix)
    similarity_matrix = cosine_similarity(sparse_count_matrix, sparse_count_matrix[movie_index])
    
    liste_similarity_matrix = similarity_matrix.flatten()
    df_similarity_matrix = pd.Series(liste_similarity_matrix)
    df_similarity_matrix = df_similarity_matrix.sort_values(ascending=False).head(6)
    
    print(df_similarity_matrix.index)
    
    df_similarity_matrix = df_similarity_matrix.drop(df_similarity_matrix.index[0])

    print(df_similarity_matrix.index)
    
    result_label.config(text=f"Voici les propositions pour : {df.iloc[movie_index]['primaryTitle']}")
    
    # Supprimer les labels précédents des propositions
    for widget in frame.winfo_children():
        widget.destroy()
    
    z=0
    for i in df_similarity_matrix.index:
        z=z+1
        label = tk.Label(frame, text=f"Film numéro {z} : {df.iloc[i]['primaryTitle']}")
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


#--------------------------------------------------------------------- Main FIN


