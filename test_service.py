import unittest
from service import (
    get_movie_details,
    get_top_rated_movies,
    get_recent_released_movies,
    search_movie,
    get_same_genres_movies,
    get_metrics
)
import datetime


class TestMovieService(unittest.TestCase):
    
    def test_get_movie_details_existing(self):
        """Test obtener detalles de película existente (The Godfather)"""
        result = get_movie_details(238)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["_id"], 238)
        self.assertEqual(result["title"], "The Godfather")
        self.assertIn("Drama", result["genres"])
        self.assertIn("Crime", result["genres"])
        self.assertIsInstance(result["vote_average"], (int, float))
        self.assertIsInstance(result["vote_count"], int)
        self.assertIsInstance(result["release_date"], datetime.datetime)
    
    def test_get_movie_details_nonexistent(self):
        """Test película que no existe"""
        result = get_movie_details(999999999)
        self.assertIsNone(result)
    
    def test_get_top_rated_movies(self):
        """Test top 25 películas mejor valoradas"""
        result = get_top_rated_movies()
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 25)
        self.assertGreater(len(result), 0)
        
        # Verificar estructura
        movie = result[0]
        self.assertIn("_id", movie)
        self.assertIn("title", movie)
        self.assertIn("vote_average", movie)
        self.assertGreaterEqual(movie["vote_count"], 5000)
        
        # Verificar orden descendente por vote_average
        for i in range(len(result) - 1):
            self.assertGreaterEqual(
                result[i]["vote_average"],
                result[i + 1]["vote_average"]
            )
    
    def test_get_recent_released_movies(self):
        """Test películas recientes con mínimo 50 votos"""
        result = get_recent_released_movies()
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 25)
        
        if result:
            movie = result[0]
            self.assertGreaterEqual(movie["vote_count"], 50)
            # Verificar que release_date es reciente (últimos ~3 años)
            cutoff = datetime.datetime.now() - datetime.timedelta(days=1160)
            self.assertGreaterEqual(movie["release_date"], cutoff)
    
    def test_search_movie(self):
        """Test búsqueda de películas por texto"""
        result = search_movie("alien")
        
        self.assertIsInstance(result, dict)
        self.assertIn("searchResults", result)
        self.assertIn("genreFacet", result)
        self.assertIn("releaseYearFacet", result)
        self.assertIn("votesFacet", result)
        
        # Verificar que hay resultados
        self.assertGreater(len(result["searchResults"]), 0)
        
        # Verificar estructura de resultado
        movie = result["searchResults"][0]
        self.assertIn("_id", movie)
        self.assertIn("title", movie)
        self.assertIn("score", movie)
        
        # Verificar que los resultados están ordenados por score (desc)
        scores = [m["score"] for m in result["searchResults"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_search_movie_empty(self):
        """Test búsqueda sin resultados"""
        result = search_movie("xyzqwertyuiop123456")
        
        self.assertEqual(len(result["searchResults"]), 0)
        self.assertEqual(len(result["genreFacet"]), 0)
    
    def test_get_same_genres_movies(self):
        """Test películas del mismo género"""
        # The Godfather tiene Drama y Crime
        result = get_same_genres_movies(238, ["Drama", "Crime"])
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 8)
        
        if result:
            # Verificar que ninguna es The Godfather
            ids = [m["_id"] for m in result]
            self.assertNotIn(238, ids)
            
            # Verificar estructura
            movie = result[0]
            self.assertIn("genres", movie)  # número de géneros coincidentes
            self.assertGreaterEqual(movie["vote_count"], 500)
            
            # Verificar orden: primero por número de géneros, luego por rating
            for i in range(len(result) - 1):
                if result[i]["genres"] == result[i + 1]["genres"]:
                    self.assertGreaterEqual(
                        result[i]["vote_average"],
                        result[i + 1]["vote_average"]
                    )
    
    def test_get_metrics(self):
        """Test obtención de métricas de rendimiento"""
        # Ejecutar algunas funciones para generar métricas
        get_movie_details(238)
        get_top_rated_movies()
        
        result = get_metrics(["get_movie_details", "get_top_rated_movies"])
        
        self.assertIsInstance(result, dict)
        self.assertIn("get_movie_details", result)
        self.assertIn("get_top_rated_movies", result)
        
        # Verificar formato de tuplas (p90, p95)
        for metric_name, (p90, p95) in result.items():
            self.assertIsInstance(p90, float)
            self.assertIsInstance(p95, float)
            self.assertGreaterEqual(p95, p90)
            self.assertGreaterEqual(p90, 0.0)


class TestMovieServicePerformance(unittest.TestCase):
    """Tests de rendimiento básicos"""
    
    def test_get_movie_details_performance(self):
        """Verificar que get_movie_details es rápido"""
        import time
        
        start = time.time()
        result = get_movie_details(238)
        elapsed = time.time() - start
        
        self.assertIsNotNone(result)
        self.assertLess(elapsed, 0.1)  # Debe ser < 100ms
    
    def test_search_movie_performance(self):
        """Verificar que search_movie es razonablemente rápido"""
        import time
        
        start = time.time()
        result = search_movie("action")
        elapsed = time.time() - start
        
        self.assertGreater(len(result["searchResults"]), 0)
        self.assertLess(elapsed, 2.0)  # Debe ser < 2 segundos


if __name__ == "__main__":
    unittest.main(verbosity=2)