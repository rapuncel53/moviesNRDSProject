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

#How to import the movies.json to the mongodb database
"""
Basic command syntax:
mongoimport --db <database_name> --collection <collection_name> --file <path_to_file> [options]

Example command
if is one document per line :
    mongoimport --db moviesdb --collection movies --file ./movies.json/TMDB_dataset_with_plot.json 

"""

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # conexion to the mongodb
db = client['moviesdb']  # select the database
collection = db['movies']  # select the collection

# Connect to Redis (dont decode responses to get binary data)
r = redis.Redis(host="localhost", port=6379, decode_responses=False)

# Connect to Neo4j
from neo4j import GraphDatabase
from collections import Counter
from typing import List, Dict, Any

uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"
driver = GraphDatabase.driver(uri, auth=(user, password))

#global variable to store metrics
stored_metrics = {}

# DO NOT MODIFY THIS FUNCTION
def measure(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            store_metric(func.__name__, end - start)

    return wrapper


@measure
def search_movie(text):
    """
    Search movies by title and sort the results using a custom score calculated as `textScore * popularity`.
    Also return facets for field `genre`, `releaseYear` and `votes`.

    Hint: check MongoDB's $facet stage
    """
    #I created the text index on title field but it should be created only once
    #so i commented it out
    if "title_text_index" not in collection.index_information():
        collection.create_index([("title", "text")], name="title_text_index")  # create text index on title field

    pipeline = [
        {"$match": {"$text": {"$search": text}}},
        {"$addFields": {"score": {"$multiply": [{"$meta": "textScore"}, "$popularity"]}}},
        {"$sort": {"score": -1}},
        {"$limit": 25},

        #$facet runs multiple sub-pipelines in parallel on the same matched input and returns one single document 
        #where each facet name is an array with that sub-pipeline’s results.
        {"$facet": {
            "genreFacet": [
                {"$unwind": "$genres"},
                {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
                {"$sort": {"count": 1}},
                
            ],
            "releaseYearFacet": [
                {"$group": {"_id": {"$year": "$release_date"}, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}},
                
            ],
            "searchResults": [
                #{"$sort": {"score": -1}},
                {"$project": {
                    "_id": 1, "title": 1, "poster_path": 1, "release_date": 1,
                    "vote_average": 1, "vote_count": 1, "score": 1
                }}
            ],
            "votesFacet": [
                {"$bucket": {
                    "groupBy": "$vote_average",
                    "boundaries": [0, 5, 7, 8, 10],
                    "default": "Other",
                    "output": {"count": {"$sum": 1}}
                }},
                {"$sort": {"_id": 1}},
            ]}}
    ]
    pipeline = collection.aggregate(pipeline)  # execute the aggregation
    pipeline = list(pipeline)
    if not pipeline:
        return {"genreFacet": [], "releaseYearFacet": [], "searchResults": [], "votesFacet": []}
    return pipeline[0]
    

"""
    return {
        "genreFacet": [
            {"_id": "Science Fiction", "count": 21},
            {"_id": "Horror", "count": 10},
            {"_id": "Action", "count": 9},
            # ...
            {"_id": "Drama", "count": 1},
        ],
        "releaseYearFacet": [
            {"_id": 1979, "count": 1},
            {"_id": 1986, "count": 1},
            # ...
            {"_id": 2022, "count": 1},
            {"_id": 2023, "count": 2},
        ],
        "searchResults": [
            {
                "_id": 981314,
                "poster_path": "/kaSvEH3RJvQa6NfAuEVqDMBEk5E.jpg",
                "release_date": datetime.datetime(2023, 5, 11, 0, 0),
                "score": 0.75,
                "title": "Alien Invasion",
                "vote_average": 5.542,
                "vote_count": 48,
            },
            {
                "_id": 126889,
                "poster_path": "/zecMELPbU5YMQpC81Z8ImaaXuf9.jpg",
                "release_date": datetime.datetime(2017, 5, 9, 0, 0),
                "score": 0.75,
                "title": "Alien: Covenant",
                "vote_average": 6.1,
                "vote_count": 7822,
            },
            # ...
        ],
        "votesFacet": [
            {"_id": 0, "count": 2},
            {"_id": 5, "count": 18},
            {"_id": 7, "count": 4},
            {"_id": 8, "count": 1},
        ],
    }
"""

@measure
def get_top_rated_movies():
    """
    Return top rated 25 movies with more than 5k votes
    """
    # Find movies with more than 5000 votes, sort by vote_average descending, limit to 25
    #top_movies = collection.find({"vote_count": {"$gt": 5000}}, {"_id": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}).sort("vote_average", -1).limit(25)
    
    #I created the index on vote_count field but it should be created only once
    if "vote_count_index" not in collection.index_information():
        collection.create_index([("vote_count", 1)], name="vote_count_index")  # create index on vote_count field

    #Using a pipeline to order the output fields
    top_movies = collection.aggregate ([
        {"$match": {"vote_count": {"$gt": 5000}}},
        {"$project": {"_id": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}},
        {"$sort": {"vote_average": -1}},
        {"$limit": 25}
    ])

    result = []
    for movie in top_movies:
        result.append({"_id": movie.get("_id"),
                      "poster_path": movie.get("poster_path"),
                      "release_date": movie.get("release_date"),
                      "title": movie.get("title"),
                      "vote_average": movie.get("vote_average"),
                      "vote_count": movie.get("vote_count")})
    return result


"""
    return [
        {
            "_id": 238,
            "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
            "release_date": datetime.datetime(1972, 3, 14, 0, 0),
            "title": "The Godfather",
            "vote_average": 8.707,
            "vote_count": 18677,
        },
        {
            "_id": 278,
            "poster_path": "/lyQBXzOQSuE59IsHyhrp0qIiPAz.jpg",
            "release_date": datetime.datetime(1994, 9, 23, 0, 0),
            "title": "The Shawshank Redemption",
            "vote_average": 8.702,
            "vote_count": 24649,
        },
    ]
"""
    
    

@measure
def get_recent_released_movies():
    """
    Return recently released movies that at least are reviewed by 50 users
    """

    #collection.create_index([("release_date", 1)])  # create index on release_date field
    #collection.create_index([("vote_count", 1)])  # create index on vote_count

    #I created the index on release_date and vote_count fields but it should be created only once
    if "release_date_vote_count_index" not in collection.index_information():
        collection.create_index([("release_date", -1), ("vote_count", -1)], name="release_date_vote_count_index")  # compound index on release_date and vote_count
   
    # find the movies that have been released in the last x days and have at least 50 votes
    recent_movies = collection.aggregate([
        {"$match": {
            "release_date": {"$gte": datetime.datetime.now() - datetime.timedelta(days=1160)}, 
            "vote_count": {"$gte": 50}}}, 
        #{"$match": {"release_date": {"$gte": ISODate("2023-01-29T00:00:00Z")}}}, 
        {"$project": {"_id": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}},
        {"$sort": {"release_date": -1}},
        {"$limit": 25}
    ])
    
    result = []
    for movie in recent_movies:
        result.append({"_id": movie.get("_id"),
                       "poster_path": movie.get("poster_path"),
                       "release_date": movie.get("release_date"),
                        "title": movie.get("title"),
                        "vote_average": movie.get("vote_average"),
                        "vote_count": movie.get("vote_count")})
    return result
    

    """
    return [
        {
            "_id": 1151534,
            "poster_path": "/rpzFxv78UvYG5yQba2soO5mMl4T.jpg",
            "release_date": datetime.datetime(2023, 9, 29, 0, 0),
            "title": "Nowhere",
            "vote_average": 7.895,
            "vote_count": 195,
        },
        {
            "_id": 866463,
            "poster_path": "/soIgqZBoTiTgMqUW0JtxsPWAilQ.jpg",
            "release_date": datetime.datetime(2023, 9, 29, 0, 0),
            "title": "Reptile",
            "vote_average": 7.354,
            "vote_count": 65,
        },
    ]
    """
    


@measure
def get_movie_details(movie_id):
    """
    Return detailed information for the specified movie_id
    """

    # Find the movie in the collection and return the required fields
    #doc = collection.find_one({"_id": movie_id}, {"genre": 1, "overview": 1, "poster_path": 1, "release_date": 1, "tagline": 1, "title": 1, "vote_average": 1, "vote_count": 1})

    #I created the index on _id field but it should be created only once
    #check if the index exists before creating it
    if "_id_1" not in collection.index_information():  
        collection.create_index([("_id", 1)], name="_id_index")  # create index on _id field



    #with a pipeline to order the output fields
    movie = collection.find_one(
        {"_id": movie_id},
        {"genres": 1, "overview": 1, "poster_path": 1, "release_date": 1, "tagline": 1, "title": 1, "vote_average": 1, "vote_count": 1}
    )

    if not movie:
        return None

    movie_details = {
        "_id": movie.get("_id"),
        "genres": movie.get("genres"),
        "overview": movie.get("overview"),
        "poster_path": movie.get("poster_path"),
        "release_date": movie.get("release_date"),
        "tagline": movie.get("tagline"),
        "title": movie.get("title"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
    }
    return movie_details
    """
    return {
        "_id": 238,
        "genres": ["Drama", "Crime"],
        "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional "
        "Italian-American Corleone crime family. When organized crime "
        "family patriarch, Vito Corleone barely survives an attempt on "
        "his life, his youngest son, Michael steps in to take care of the "
        "would-be killers, launching a campaign of bloody revenge.",
        "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
        "release_date": datetime.datetime(1972, 3, 14, 0, 0),
        "tagline": "An offer you can't refuse.",
        "title": "The Godfather",
        "vote_average": 8.707,
        "vote_count": 18677,
    }"""
   


@measure
def get_same_genres_movies(movie_id, genres):
    """
    Return a list of movies that match at least one of the provided genres.

    Movies need to be sorted by the number genres that match in descending order
    (a movie matching two genres will appear before a movie only matching one). When
    several movies match with the same number of genres, movies with greater rating must
    appear first.

    Discard movies with votes by less than 500 users. Limit to 8 results.
    """
    #create indexes to optimize the query, but should be created only once
    if "genres_1_vote_count_1_id_1" not in collection.index_information():
        collection.create_index([("genres",1), ("vote_count",1), ("_id",1)], name="genres_1_vote_count_1_id_1")  # create compound index on genres and vote_count
  
    same_genre_movies = collection.aggregate ([
        {"$match": {"genres": {"$in": genres}, "vote_count": {"$gte": 500}, "_id": {"$ne": movie_id}}},
        {"$addFields": {"genres": {"$size": {"$setIntersection": ["$genres", genres]}}}},
        {"$project": {"_id": 1,"genres": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}},
        {"$sort": {"genres": -1, "vote_average": -1}},
        {"$limit": 8}
    ])

    result = []
    for movie in same_genre_movies:
        result.append({"_id": movie.get("_id"),
                       "genres": movie.get("genres"),
                       "poster_path": movie.get("poster_path"),
                       "release_date": movie.get("release_date"),
                        "title": movie.get("title"),
                        "vote_average": movie.get("vote_average"),
                        "vote_count": movie.get("vote_count")})
    return result


    """
    return [
        {
            "_id": 335,
            "genres": 2,
            "poster_path": "/qbYgqOczabWNn2XKwgMtVrntD6P.jpg",
            "release_date": datetime.datetime(1968, 12, 21, 0, 0),
            "title": "Once Upon a Time in the West",
            "vote_average": 8.294,
            "vote_count": 3923,
        },
        {
            "_id": 3090,
            "genres": 2,
            "poster_path": "/pWcst7zVbi8Z8W6GFrdNE7HHRxL.jpg",
            "release_date": datetime.datetime(1948, 1, 15, 0, 0),
            "title": "The Treasure of the Sierra Madre",
            "vote_average": 7.976,
            "vote_count": 1066,
        },
    ]

    """
 

@measure
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

    # obtain the embedding and conver to binary data
    embedding_bytes = data[b"embedding"]
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

    # Query KNN: buscar los 5 más similares (excluyendo el propio movie_id)
    q = (
        Query("*=>[KNN 6 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("movie_id", "title", "score")
        .dialect(2) 
    )

    # Ejecutar busqueda en Redis
    results = r.ft("movie_idx").search(q, query_params={"vec": embedding.tobytes()})

    # Recoger los movie_ids (excepto el mismo)
    similar_ids = []
    for doc in results.docs:
        mid = doc.movie_id
        if mid != str(movie_id):
            similar_ids.append(mid)
        if len(similar_ids) >= 5:
            break

    #I created the index on _id field but it should be created only once
    if "_id_index" not in collection.index_information():
        collection.create_index([("_id", 1)], name="_id_index")  # create index on _id field

    # Buscar detalles en MongoDB
    movies = list(collection.find(
        {"_id": {"$in": [int(mid) for mid in similar_ids]}},
        {"_id": 1, "genres": 1, "poster_path": 1, "release_date": 1, "title": 1, "vote_average": 1, "vote_count": 1}

    ).sort("popularity", -1))

    # Ordenar por popularidad (popularity descendente)
    # movies = sorted(movies, key=lambda m: m.get("popularity", 0), reverse=True)

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
    
    """
    return [
        {
            "_id": 335,
            "genres": 2,
            "poster_path": "/qbYgqOczabWNn2XKwgMtVrntD6P.jpg",
            "release_date": datetime.datetime(1968, 12, 21, 0, 0),
            "title": "Once Upon a Time in the West",
            "vote_average": 8.294,
            "vote_count": 3923,
        },
        {
            "_id": 3090,
            "genres": 2,
            "poster_path": "/pWcst7zVbi8Z8W6GFrdNE7HHRxL.jpg",
            "release_date": datetime.datetime(1948, 1, 15, 0, 0),
            "title": "The Treasure of the Sierra Madre",
            "vote_average": 7.976,
            "vote_count": 1066,
        },
    ]
"""

@measure
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
        """
        usernames = []
        for record in result:
            usernames.append(record["username"])
        """

        # Imprimir cada username en una linea
        for u in usernames:
            print(u)

        return usernames
        


@measure
def get_recommendations_for_user(username):
    """
    Return up to 10 movies based on similar users taste.
    """
    with driver.session() as session:
        # Obtener peliculas que el usuario ha dado like
        user_movies_result = session.run("""
            MATCH (u:Usuario {username: $username})-[:FAVORITE]->(m:Pelicula)
            RETURN collect(m.id) AS movies
        """, username=str(username)).single()

        if not user_movies_result or not user_movies_result["movies"]:
            print(" El usuario no ha dado 'like' a ninguna película.")
            return []


        user_movies = user_movies_result["movies"]


        # Encontrar otros usuarios con al menos una pelicula en comun
        neighbors_result = session.run("""
            MATCH (u1:Usuario {username: $username})-[:FAVORITE]->(m:Pelicula)<-[:FAVORITE]-(u2:Usuario)
            WHERE u1 <> u2
            WITH DISTINCT u2
            MATCH (u2)-[:FAVORITE]->(m2:Pelicula)
            RETURN u2.username AS neighbor, collect(m2.id) AS movies
        """, username=str(username))
        #encuentra dos usuarios con la misma pelicula favorita
        #que sean diferentes 
        #que no se repitan (devuelve distinos u2)
        #luego busca las peliculas favoritas de esos usuarios
        #devuelve el nombre del usuario vecino y las peliculas favoritas de ese usuario

        # Calcular similitud Jaccard entre el usuario y sus vecinos
        similarities = []
        for record in neighbors_result:
            neighbor_movies = set(record["movies"])
            intersection = len(set(user_movies) & set(neighbor_movies))
            union = len(set(user_movies) | set(neighbor_movies))
            if union > 0:
                jaccard = intersection / union
                similarities.append((record["neighbor"], jaccard, neighbor_movies))

        if not similarities:
            print(" No se encontraron usuarios similares.")
            return []

        # Elegir los K vecinos más parecidos
        k = 5
        top_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        #ordenar el array de similitudes por el valor de jaccard (el segundo elemento)y quedarse con los k primeros

        # Contar peliculas que les gustaron a los vecinos pero el usuario no ha visto
        recommended = Counter()
        for neighbor, score, neighbor_movies in top_neighbors:
            for m in neighbor_movies:
                if m not in user_movies:
                    recommended[m] += 1
        # Recorre las películas favoritas de los vecinos más similares
        # y cuenta cuántas veces aparece cada película que el usuario aún no ha visto.
        # Esto sirve para identificar qué películas son más recomendadas según los gustos de otros usuarios.


        if not recommended:
            print("No hay recomendaciones nuevas.")
            return []

        # Obtener detalles de las peliculas recomendadas (suponiendo que existen mas propiedades)
        top_movie_ids = [m for m, _ in recommended.most_common(10)] # obtener los ids de las 10 peliculas mas recomendadas
        print(f"Top recommended movie IDs for user {username}: {top_movie_ids}")

        #print(f"Top recommended movie IDs for user {username}: {top_movie_ids}")

        top_movie_ids = [int(i) for i in top_movie_ids]
        
        # Buscar detalles en MongoDB
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
"""
    return [
        {
            "_id": 496243,
            "poster_path": "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
            "release_date": datetime.datetime(2019, 5, 30, 0, 0),
            "title": "Parasite",
            "vote_average": 8.515,
            "vote_count": 16430,
        }
    ]
"""

#you have to uncomment the index lines to get the performance over 75ms

def get_metrics(metrics_names: List) -> Dict[str, Tuple[float, float]]:
    """
    Return 95th percentile in seconds for each one of the given metric names
    """
    global stored_metrics
    results = {}
    # I understand that here I have to return just the 95th percentile of each metric name.
    for metric in metrics_names:
        # calculate percentiles for each metric
        if metric in stored_metrics:
            times = stored_metrics[metric]

            percentile_90 = np.percentile(times, 90)
            percentile_95 = np.percentile(times, 95)
            results[metric] = (percentile_90, percentile_95)
        else:
            results[metric] = (0.0, 0.0)

    return results

    #return {name: (0.9, 0.95) for name in metrics_names}


def store_metric(metric_name: str, measure_s: float):
    """
    Store mesured sample in seconds of the given metric
    """
    global stored_metrics
    if metric_name not in stored_metrics:
        stored_metrics[metric_name] = []
    # Añadimos el valor

    # I understand that this gives me the metric name of some specific function and the time that it took to execute it.
    stored_metrics[metric_name].append(measure_s)

    pass

#if __name__ == "__main__":
    