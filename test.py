from pymongo import MongoClient
import redis
from sentence_transformers import SentenceTransformer
import datetime
import pickle
#create the index for the plot embedding
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from typing import Dict, List, Tuple
from pymongo import MongoClient
from collections import OrderedDict
import datetime
import time
import statistics
import numpy as np
import pickle
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from bson import ObjectId

# Connect to Neo4j
from neo4j import GraphDatabase
from collections import Counter
from typing import List, Dict, Any

uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"
driver = GraphDatabase.driver(uri, auth=(user, password))


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # conexion to the mongodb
db = client['moviesdb']  # select the database
collection = db['movies']  # select the collection

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=False)

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")

INDEX_NAME = "movie_idx"
DOC_PREFIX = "movie:"

def get_similar_movies(movie_id):
    """
    Return a list of movies with a similar plot as the given movie_id.
    Movies need to be sorted by the popularity instead of proximity score.
    """

    # Obtain the hash with the embedding from Redis from that movie_id
    key = f"movie:{movie_id}"
    data = r.hgetall(key)
    if not data:
        print("Movie not found in Redis")
        return []

    # Obtener el embedding (lo guardamos en binario, así que hay que convertirlo)
    embedding_bytes = data[b"embedding"]
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    print(len(embedding))
    print(f"Embedding for movie ID {movie_id} retrieved.")
    print(f"Embedding vector (first 5 values): {embedding[:5]}")
    print(r.ft("movie_idx").info())

    # Query KNN: buscar los 5 más similares (excluyendo el propio movie_id)
    q = (
        Query("*=>[KNN 6 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("movie_id", "title", "score")
        .dialect(2)
    )

    # Ejecutar búsqueda en Redis
    results = r.ft("movie_idx").search(q, query_params={"vec": embedding.tobytes()})
    print(f"Found {results.total} similar movies for movie ID {movie_id}.")
    # Recoger los movie_ids (excepto el mismo)
    similar_ids = []
    for doc in results.docs:
        mid = doc.movie_id
        if mid != str(movie_id):
            similar_ids.append(mid)
        if len(similar_ids) >= 5:
            break

    # Buscar detalles en MongoDB
    movies = list(collection.find(
        {"_id": {"$in": [int(mid) for mid in similar_ids]}},
        {"_id": 1, "genres": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}
    ))

    # Ordenar por popularidad (popularity descendente)
    movies = sorted(movies, key=lambda m: m.get("popularity", 0), reverse=True)

    resultss = []
    for movie in movies:
        resultss.append({"_id": movie.get("_id"),
                       "genres": movie.get("genres"),
                       "poster_path": movie.get("poster_path"),
                       "release_date": movie.get("release_date"),
                        "title": movie.get("title"),
                        "vote_average": movie.get("vote_average"),
                        "vote_count": movie.get("vote_count")})
    return resultss

def get_movie_likes(username, movie_id):
    """
    Returns a list of usernames of users who also like the specified movie_id
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:Usuario)-[:FAVORITE]->(m:Pelicula {id: $movie_id})
            RETURN u.username AS username
            LIMIT 100
            """,
            movie_id=str(movie_id)
        )
        
        # Materializar resultados en lista
        usernames = [record["username"] for record in result]

        # Imprimir cada username en una línea
        for u in usernames:
            print(u)

        return usernames

def get_recommendations_for_user(username):
    """
    Return up to 10 movies based on similar users taste.
    """
    with driver.session() as session:
        # 1️⃣ Obtener películas que el usuario ha dado "like"
        user_movies_result = session.run("""
            MATCH (u:Usuario {username: $username})-[:FAVORITE]->(m:Pelicula)
            RETURN collect(m.id) AS movies
        """, username=str(username)).single()

        if not user_movies_result or not user_movies_result["movies"]:
            print("❌ El usuario no ha dado 'like' a ninguna película.")
            return []


        user_movies = user_movies_result["movies"]


        # 2️⃣ Encontrar otros usuarios con al menos una película en común
        neighbors_result = session.run("""
            MATCH (u1:Usuario {username: $username})-[:FAVORITE]->(m:Pelicula)<-[:FAVORITE]-(u2:Usuario)
            WHERE u1 <> u2
            WITH DISTINCT u2
            MATCH (u2)-[:FAVORITE]->(m2:Pelicula)
            RETURN u2.username AS neighbor, collect(m2.id) AS movies
        """, username=str(username))

        similarities = []
        for record in neighbors_result:
            neighbor_movies = set(record["movies"])
            intersection = len(set(user_movies) & set(neighbor_movies))
            union = len(set(user_movies) | set(neighbor_movies))
            if union > 0:
                jaccard = intersection / union
                similarities.append((record["neighbor"], jaccard, neighbor_movies))

        if not similarities:
            print("⚠️ No se encontraron usuarios similares.")
            return []

        # 3️⃣ Elegir los K vecinos más parecidos
        k = 5
        top_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

        # 4️⃣ Contar películas que les gustaron a los vecinos pero el usuario no ha visto
        recommended = Counter()
        for neighbor, score, neighbor_movies in top_neighbors:
            for m in neighbor_movies:
                if m not in user_movies:
                    recommended[m] += 1

        if not recommended:
            print("⚠️ No hay recomendaciones nuevas.")
            return []

        # 5️⃣ Obtener detalles de las películas recomendadas (suponiendo que existen más propiedades)
        top_movie_ids = [m for m, _ in recommended.most_common(10)]
        print(f"Top recommended movie IDs for user {username}: {top_movie_ids}")

        #print(f"Top recommended movie IDs for user {username}: {top_movie_ids}")

        top_movie_ids = [int(i) for i in top_movie_ids]
        
        # 6️⃣ Buscar detalles en MongoDB
        movies_cursor = collection.find(
            {"_id": {"$in": top_movie_ids}},
            {"_id": 1, "poster_path": 1, "release_date": 1, "title": 1,
             "vote_average": 1, "vote_count": 1}
        )

        result = []
        for movie in movies_cursor:
            result.append({
                "_id": movie.get("_id"),
                "poster_path": movie.get("poster_path"),
                "release_date": movie.get("release_date") or datetime.datetime(1900, 1, 1),
                "title": movie.get("title"),
                "vote_average": movie.get("vote_average", 0.0),
                "vote_count": movie.get("vote_count", 0)
            })
        print(f"Recommended movies for user {username}: {result}")
        return result

if __name__ == "__main__":
    
    get_recommendations_for_user("julian.bascones")
    #get_movie_likes("user123", 157336)
    #movie_id = 597  # Example movie ID
    #similar_movies = get_similar_movies(movie_id)
    #print(f"Similar movies to movie ID {movie_id}:")
    #for movie in similar_movies:
        #print(movie)