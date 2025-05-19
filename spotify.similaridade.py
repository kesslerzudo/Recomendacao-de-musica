# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:59:08 2025

@author: anton
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('spotify_dataset.csv')

print("Visualizando as primeiras linhas do dataset:")
print(df.head())

features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

for feature in features:
    if feature not in df.columns:
        print(f"A característica '{feature}' não está presente no dataset.")

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[features])

cosine_sim = cosine_similarity(normalized_features)

indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()
 
def generate_recommendations(song_title, cosine_sim=cosine_sim):
    """
    Gera uma lista de recomendações de músicas semelhantes à música fornecida.

    Parâmetros:
    - song_title: título da música de referência
    - cosine_sim: matriz de similaridade do cosseno

    Retorna:
    - Lista de títulos das 10 músicas mais semelhantes
    """
    if song_title not in indices:
        print(f"A música '{song_title}' não foi encontrada no dataset.")
        return []

    idx = indices[song_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    song_indices = [i[0] for i in sim_scores]

    return df['song_title'].iloc[song_indices]

song = input("Digite o título de uma música para obter recomendações: ")

recommendations = generate_recommendations(song)

if recommendations:
    print(f"\nMúsicas semelhantes a '{song}':\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("Não foi possível gerar recomendações.")
