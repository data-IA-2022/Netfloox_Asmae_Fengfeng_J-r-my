import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def model (film, df_recommend):

    grouped = df_recommend.groupby(['primaryTitle', 'category'])['primaryName'].apply(lambda x: ', '.join(x[:3]))
    data_pivoted = grouped.unstack()
    df_recommend.drop(['category', 'primaryName'],axis=1, inplace=True)
    df=pd.merge(df_recommend, data_pivoted, on='primaryTitle')

    def concat_features(row):
        return(str(row['genres']).replace(",", " ") + " " + str(row['startYear']).replace(" ", "") + " " + str(row['actor']).replace(" ", "") + " " +
            str(row["actress"]).replace(" ", "") + " " + str(row["cinematographer"]).replace(" ", "") + " " + str(row["composer"]).replace(" ", "") + " " +
            str(row["director"]).replace(" ", "") + " " + str(row["editor"]).replace(" ", "") + " " + str(row["producer"]).replace(" ", "") + " " +
            str(row["production_designer"]).replace(" ", "") + " " + str(row["self"]).replace(" ", "") + " " + str(row["writer"]).replace(" ", "") + " " +
            str(row["archive_footage"]).replace(" ", ""))

    df = df.replace('\\N', 'nan')
    df["movie_features"] = df.apply(concat_features, axis=1)
    df = df.drop_duplicates(subset=['movie_features'])
    print(df.head(50))

    text = film
    
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

    z=0
    recommandations = list()
    for i in df_similarity_matrix.index:
        z=z+1
        #print(f"Film numéro {z} : {df.iloc[i]['primaryTitle']}")
        recommandations.append(df.iloc[i]['primaryTitle'])

    return recommandations #(df_similarity_matrix)