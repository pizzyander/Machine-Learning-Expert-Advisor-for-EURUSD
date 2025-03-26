from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from keras.saving import register_keras_serializable
import joblib
import sklearn

# Register custom loss function if needed
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
MODEL_PATH = "models/gru_model.keras"
scaler = "models/y_scaler.pkl"
# Load the trained model with custom objects
custom_objects = {'mse': mse}
model = tf.keras.models.load_model(MODEL_PATH , custom_objects=custom_objects)
y_scaler = joblib.load(scaler)

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # Convert input data to NumPy array
        features = np.array(data.features, dtype=np.float32)

        # Ensure input is reshaped correctly
        features = features.reshape(1, 30, 18)  # Match model's expected input shape
        
        # Perform inference
        prediction = model.predict(features)
        
        # Inverse transform the prediction
        prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1))  # Ensure correct shape

        # Return the prediction result
        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def home():
    return {"message": "Forex LSTM Model API is running!"}
