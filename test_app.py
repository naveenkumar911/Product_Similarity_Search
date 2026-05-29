import unittest
import time

# Monkey-patch httpx.Client to fix starlette 0.35.1 and httpx 0.28.1 incompatibility
import httpx
_orig_client_init = httpx.Client.__init__
def _patched_client_init(self, *args, **kwargs):
    kwargs.pop("app", None)
    return _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client_init

from fastapi.testclient import TestClient
from app import app
from similarity import find_similar_products, df_products, id_to_idx

class TestProductSimilarityAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the test client
        cls.client = TestClient(app)
        # Use a known product ID from the dataset
        cls.valid_product_id = "26d41bdc1495de290bc8e6062d927729"
        cls.invalid_product_id = "nonexistent_product_id_12345"
        
        # Verify indices are loaded
        if df_products is None:
            raise RuntimeError("Data and index failed to load in similarity.py")
            
    def test_root_endpoint(self):
        """Test the index root returns basic service metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "Product Similarity Search API")
        self.assertEqual(data["status"], "healthy")
        self.assertGreater(data["total_indexed_products"], 0)

    def test_health_endpoint(self):
        """Test the health check endpoint for deployment compatibility."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")
        self.assertIn("timestamp", data)

    def test_get_product_valid(self):
        """Test retrieving details of a valid product."""
        response = self.client.get(f"/products/{self.valid_product_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uniq_id"], self.valid_product_id)
        self.assertIn("product_name", data)
        self.assertIn("sales_price", data)
        self.assertIn("image_ids", data)

    def test_get_product_invalid(self):
        """Test retrieving details of a non-existent product returns 404."""
        response = self.client.get(f"/products/{self.invalid_product_id}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

    def test_find_similar_products_success(self):
        """Test basic successful similarity search retrieval."""
        num_similar = 5
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar={num_similar}"
        )
        self.assertEqual(response.status_code, 200)
        similar_ids = response.json()
        
        self.assertEqual(len(similar_ids), num_similar)
        # The query product itself must not be in the output
        self.assertNotIn(self.valid_product_id, similar_ids)
        # All items must be valid unique IDs
        for uniq_id in similar_ids:
            self.assertIn(uniq_id, id_to_idx)

    def test_find_similar_products_invalid_id(self):
        """Test that searching with an invalid product ID returns 404."""
        response = self.client.get(
            f"/find_similar_products?product_id={self.invalid_product_id}&num_similar=5"
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

    def test_find_similar_products_bounds(self):
        """Test parameter boundary validation (num_similar bounds)."""
        # Test num_similar too small (0)
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=0"
        )
        self.assertEqual(response.status_code, 422)  # FastAPI Query validation error
        
        # Test num_similar too large (101)
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=101"
        )
        self.assertEqual(response.status_code, 422)

    def test_find_similar_products_weights_validation(self):
        """Test similarity weights constraints and validation."""
        # Test negative weight
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=5&w_text=-0.5"
        )
        self.assertEqual(response.status_code, 422)
        
        # Test weights sum up to 0.0
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=5&w_text=0.0&w_brand=0.0&w_image=0.0&w_price=0.0&w_rating=0.0&w_weight=0.0"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("weights must be greater than zero", response.json()["detail"])

    def test_custom_weights_effect(self):
        """Verify that altering weights changes ranking output, confirming weights are utilized."""
        # Query prioritising text vs query prioritising brand
        res_text_heavy = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=5&w_text=1.0&w_brand=0.0&w_price=0.0"
        )
        res_brand_heavy = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=5&w_text=0.0&w_brand=1.0&w_price=0.0"
        )
        
        self.assertEqual(res_text_heavy.status_code, 200)
        self.assertEqual(res_brand_heavy.status_code, 200)
        
        # Verify similarity list outputs are different due to weighting difference
        self.assertNotEqual(res_text_heavy.json(), res_brand_heavy.json())

    def test_search_performance(self):
        """Test search query response time to ensure latency is within enterprise SLAs (<100ms)."""
        start_time = time.time()
        response = self.client.get(
            f"/find_similar_products?product_id={self.valid_product_id}&num_similar=10"
        )
        duration_ms = (time.time() - start_time) * 1000
        
        self.assertEqual(response.status_code, 200)
        # Search latency SLA check (<100ms)
        self.assertLess(duration_ms, 100.0, f"Query took {duration_ms:.2f}ms, which is above the 100ms SLA limit")
        print(f"Query performance: {duration_ms:.2f}ms")

if __name__ == "__main__":
    unittest.main()
