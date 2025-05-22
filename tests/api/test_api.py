import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, MagicMock

# We need to patch joblib.load BEFORE importing the app
import joblib
# Create the patch before importing the app
with patch('joblib.load') as mock_load:
    # Configure the mock to return a MagicMock object
    mock_model = MagicMock()
    mock_transformer = MagicMock()
    # Set up the side_effect to return different mocks for different arguments
    def load_side_effect(filename):
        if filename == 'models/model.pkl':
            return mock_model
        elif filename == 'models/transformer.pkl':
            return mock_transformer
        raise FileNotFoundError(f"Mock couldn't find {filename}")
    
    mock_load.side_effect = load_side_effect
    
    # Now import the app after the patch is in place
    from api.api import app, CustomerData

# Create the test client
client = TestClient(app)

# Mock data for testing
mock_customer_data = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Bank Customer Churn Prediction API"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    # Configure the mock model and transformer for this test
    mock_transformer.transform.return_value = [[1, 0, 0, 35, 5, 75000.0, 2, 1, 1, 50000.0]]
    mock_transformer.get_feature_names_out.return_value = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    
    response = client.post("/predict", json=mock_customer_data)
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "prediction_label" in result
    assert result["prediction"] == 0
    assert result["prediction_label"] == "Not Churned"
    assert "probability" in result
