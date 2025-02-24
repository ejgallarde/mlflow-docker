import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi_dir.app import app  # Ensure correct module import


class TestFastAPIApp(unittest.TestCase):  #Inherits unittest.TestCase; inside class for organization, lifecycle mgmt, & test discovery
    @patch("mlflow.pyfunc")  #T decorator temporarily replaces the mlflow.pyfunc module in the scope of this test function with mock object.
    def test_predict(self, mock_get_model):  #mock_get_model is the injected mock object corresponding to mlflow.pyfunc
        """
        Test the /predict endpoint.
        Purpose: Isolates the test from the actual MLflow dependency, ensuring the test doesn't attempt to load a real model
        """
        client = TestClient(app)  # Creates a simulated client to send HTTP requests to the FastAPI app
        # It allows the test to interact with the API endpoints as if they were running on a server.

        # Mock MLflow Model
        # Sets up the mock so that whenever its predict method is called, it returns the list [0, 1, 2]. This simulates the prediction output.
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 2]
        mock_get_model.return_value = mock_model

        # Send a request
        # Uses the TestClient to send a POST request to the /predict endpoint.
        # The JSON payload provided matches the expected format defined by the FastAPI endpoint
        response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2], [6.1, 2.8, 4.7, 1.2]]})

        # Assertions
        assert response.status_code == 200
        assert "predictions" in response.json()
