from pymongo import MongoClient
import redis
from sentence_transformers import SentenceTransformer
import datetime
import pickle
#create the index for the plot embedding
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


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

#The vector dimmensions are 384 for the avsolatorio/GIST-small-Embedding-v0 model
def create_index(vector_dimensions: int):
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except Exception:
        # schema of the index what types of fields are in my doccuments
        schema = (
            TextField("title"),
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": vector_dimensions,
                "DISTANCE_METRIC": "COSINE", #use the cosine distance metric
            }),
        )

        # definition of the index (usa IndexType.HASH)
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

        # crear índice
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(" Index creado correctamente")


def populate_redis():
    """
    Populate Redis with movie plot embeddings from MongoDB.
    Only include movies with at least 500 votes.
    """
    # Query MongoDB for movies with at least 500 votes
    plot = collection.aggregate([{"$match": 
        {"vote_count": {"$gte": 500}}}, 
        {"$project": {"_id": 1,"title":1, "plot": 1, "overview": 1}}])
    # Iterate through the movies and get the id,plot and overview
    for movie in plot:
        movie_id = movie["_id"]
        plot = movie.get("plot") or movie.get("overview") # Use overview if plot is not available
        if not plot:
            continue
        #create the embedding with the loaded model
        embedding = model.encode(plot, normalize_embeddings=True)
        print(f"len(embedding): {len(embedding)}")

        key = f"movie:{movie['_id']}"
        r.hset(key, mapping={
                "movie_id": str(movie["_id"]),
                "title": movie.get("title", ""),
                # Serialize the vector to binary to store it
                "embedding": embedding.astype("float32").tobytes()
        })

        print(f"Guardada película {movie.get('title')} en Redis")

if __name__ == "__main__":
    populate_redis()
    print(" Redis populated with embeddings")
    create_index(vector_dimensions=384)
    print(" Index created in Redis")