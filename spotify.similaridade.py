# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:59:08 2025

@author: anton
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('spotify_dataset.csv')  

features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence']

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[features])

similarity_matrix = cosine_similarity(normalized_features)

indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

def recomendar_musicas(titulo_musica, n_recomendacoes=10):
    if titulo_musica not in indices:
        return f"A música '{titulo_musica}' não foi encontrada no conjunto de dados."
    
    idx = indices[titulo_musica]
    similaridades = list(enumerate(similarity_matrix[idx]))
    similaridades = sorted(similaridades, key=lambda x: x[1], reverse=True)
    similaridades = similaridades[1:n_recomendacoes+1]
    musicas_recomendadas = [df['song_title'].iloc[i[0]] for i in similaridades]
    return musicas_recomendadas

musica = 'Shape of You'  
recomendacoes = recomendar_musicas(musica)
print(f"Músicas recomendadas para '{musica}':")
for i, rec in enumerate(recomendacoes, 1):
    print(f"{i}. {rec}")
