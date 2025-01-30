from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from keras.saving import register_keras_serializable

# Register custom loss function if needed
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the trained model with custom objects
custom_objects = {'mse': mse}
model = tf.keras.models.load_model("forex_lstm_model.h5", custom_objects=custom_objects)

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # Convert input data to NumPy array and reshape
        features = np.array(data.features, dtype=np.float32)
        features = features.reshape(1, features.shape[0], features.shape[1])  # Ensure correct shape for LSTM
        
        # Perform inference
        prediction = model.predict(features)
        
        # Return the prediction result
        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Forex LSTM Model API is running!"}
