from fastapi import FastAPI, Query, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import mlflow.pyfunc
import pandas as pd
from typing import Union, Optional
import time

import joblib
import logging
from pydantic import BaseModel, Field

# Add these new metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions', ['result'])
PREDICTION_LATENCY = Histogram('model_prediction_seconds', 'Time spent processing prediction', 
                              buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI()

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

try:
    logger.info("Loading model and transformer...")
    model_path = "models/model.pkl"
    transformer_path = "models/transformer.pkl"
    
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    
    logger.info("Model and transformer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")


class CustomerData(BaseModel):
    CreditScore: int = Field(..., example=650)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., example=35)
    Tenure: int = Field(..., example=5)
    Balance: float = Field(..., example=75000.0)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=50000.0)

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    prediction_label: str

@app.get("/", tags=["General"])
async def home():
    """Home endpoint that returns a welcome message."""
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Bank Customer Churn Prediction API"}

@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "model_loaded": model is not None, "transformer_loaded": transformer is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: CustomerData):
    """
    Predict customer churn based on input features.
    
    Returns:
        prediction: 0 (not churned) or 1 (churned)
        prediction_label: "Not Churned" or "Churned"
    """
    try:
        logger.info("Received prediction request")
        
        # Start timing the prediction
        start_time = time.time()
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        logger.debug(f"Input data: {input_df}")
        
        # Transform the input data
        transformed_data = transformer.transform(input_df)
        transformed_df = pd.DataFrame(transformed_data, columns=transformer.get_feature_names_out())
        logger.debug(f"Transformed data shape: {transformed_df.shape}")
        
        # Make prediction
        prediction = int(model.predict(transformed_df)[0])
        logger.info(f"Prediction result: {prediction}")
                
        # Get prediction label
        prediction_label = "Churned" if prediction == 1 else "Not Churned"
        
        # Try to get probability if the model supports it
        probability = None
        try:
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(transformed_df)[0][1])
                logger.debug(f"Prediction probability: {probability}")
        except Exception as e:
            logger.warning(f"Could not get probability: {str(e)}")
        
        # Record metrics
        prediction_time = time.time() - start_time
        PREDICTION_LATENCY.observe(prediction_time)
        PREDICTION_COUNTER.labels(result=prediction_label).inc()
        
        return {
            "prediction": prediction,
            "probability": probability,
            "prediction_label": prediction_label
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
