import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create mock objects that can be used across tests
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0]
    model.predict_proba.return_value = [[0.8, 0.2]]
    return model

@pytest.fixture
def mock_transformer():
    transformer = MagicMock()
    transformer.transform.return_value = [[1, 0, 0, 35, 5, 75000.0, 2, 1, 1, 50000.0]]
    transformer.get_feature_names_out.return_value = ['feature1', 'feature2']
    return transformer