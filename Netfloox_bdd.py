# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:45:13 2023

@author: jejel
"""
#---------------------------------------------------------------- Imports DEBUT
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine

#------------------------------------------------------------------ Imports FIN



#------------------------------------------------------------ BDD Connect DEBUT

try:  
    conn = create_engine('mysql+pymysql://netfloox:987456321@localhost/netfloox')
    print("La connexion à la base de données a été établie avec succès")
except Exception as e:
    print("Impossible de se connecter à la base de données. Raison :", str(e))
#-------------------------------------------------------------- BDD Connect FIN



#------------------------------------------------------------------- Main DEBUT
files=['name.basics', 'akas', 'basics', 'crew', 'episode', 'principals', 'ratings']

np.random.seed(123)

for name in files:
    print(f"Chargement {name}")
    df = pd.read_csv(f"C:/Users/jejel/Desktop/Netfloox/{name}.tsv", sep='\t')
    print(df.shape)
     
    # df.to_sql(name.replace('.', '_'), conn, if_exists='replace', index=False)
    
    if df.shape[0] < 5000000:
        n_samples = df.shape[0]
    else:
        n_samples = 5000000  # Number of samples to select
        
    df_sample = df.sample(n_samples)
    print(df_sample.shape)
    df_sample.to_sql(name.replace('.', '_'), conn, if_exists='replace', index=False)
#--------------------------------------------------------------------- Main FIN














