import pytest
from unittest.mock import patch, MagicMock
import json

# Import the function to test
from app.app import get_prediction

@patch('app.app.rs.post')
def test_get_prediction(mock_post):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prediction": 0,
        "probability": 0.2,
        "prediction_label": "Not Churned"
    }
    mock_post.return_value = mock_response
    
    # Test data
    test_data = {
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
    
    # Call the function
    result = get_prediction(test_data)
    
    # Assertions
    mock_post.assert_called_once_with(
        "http://api:8086/predict", 
        data=json.dumps(test_data), 
        headers={"Content-Type": "application/json"}
    )
    assert result == {"prediction": 0, "probability": 0.2, "prediction_label": "Not Churned"}