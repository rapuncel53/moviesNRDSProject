#populate_neo4j.py
import pandas as pd
from neo4j import GraphDatabase

# Conexión a Neo4j
# Cambia la URI, usuario y contraseña según tu instalación
uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Leer el CSV con pandas
df = pd.read_csv("Favorite movies 2025 (respostes) - Respostes al formulari 1.csv")


# Función para crear nodos y relaciones
def import_data(tx, username, movies):
    # Crear nodo Usuario
    tx.run(
        "MERGE (u:Usuario {username: $username})",
        username=username
    )
    # Crear nodos Película y relaciones
    for movie_id in movies:
        tx.run(
            """
            MERGE (m:Pelicula {id: $movie_id})
            MERGE (u:Usuario {username: $username})
            MERGE (u)-[:FAVORITE]->(m)
            """,
            username=username,
            movie_id=movie_id.strip()
        )

# Iterar sobre el DataFrame para poblar Neo4j
with driver.session() as session:
    for _, row in df.iterrows():
        username = row.iloc[1]  # segunda columna: usuario
        movies = str(row.iloc[2]).split(",")  # tercera columna: lista de IDs
        session.execute_write(import_data, username, movies)

driver.close()