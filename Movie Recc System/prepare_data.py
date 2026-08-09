import pandas as pd
import numpy as np
import json
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading raw CSV files...")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

print("Merging datasets...")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def process_weird_genres_keywords(obj):
    L = []
    for i in json.loads(obj):
        L.append(i['name'])
    return L

def process_weird_cast(cast):
    casts = []
    cast = json.loads(cast)
    for c in cast[:3]:
        casts.append(c['name'])
    return casts

def process_weird_crew(crew):
    crews = []
    crew = json.loads(crew)
    for c in crew:
        if c['job'] == 'Director':
            crews.append(c['name'])
    return crews

print("Cleaning columns...")
movies['genres'] = movies['genres'].apply(process_weird_genres_keywords)
movies['keywords'] = movies['keywords'].apply(process_weird_genres_keywords)
movies['cast'] = movies['cast'].apply(process_weird_cast)
movies['crew'] = movies['crew'].apply(process_weird_crew)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies_clean = movies[['movie_id', 'title', 'tags']].copy()
movies_clean['tags'] = movies_clean['tags'].apply(lambda x: " ".join(x))
movies_clean['tags'] = movies_clean['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

print("Stemming text tags...")
movies_clean['tags'] = movies_clean['tags'].apply(stem)

print("Computing TF-IDF / Count Vectors & Cosine Similarity...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_clean['tags']).toarray()
similarity = cosine_similarity(vectors).astype(np.float32)

print("Saving pickled models (movies_dict.pkl & similarity.pkl)...")
movies_dict = movies_clean[['movie_id', 'title']].to_dict(orient='records')
with open('movies_dict.pkl', 'wb') as f:
    pickle.dump(movies_dict, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f, protocol=4)

print("Model preparation completed successfully!")
