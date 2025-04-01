import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

actual = np.array([1.04588, 1.0936,  1.05262, 1.08511, 1.10923, 1.06126, 1.10967, 1.04348, 1.03016,
    1.03729, 1.08942, 1.08882, 1.07886, 1.118,   1.05068, 1.08549, 1.10325, 1.08882,
    1.04774, 1.05025])

predicted = np.array([1.0466508, 1.0927643, 1.0527341, 1.0834514, 1.108835,  1.0639707, 1.110892,
    1.041937, 1.0309708, 1.0379746, 1.0877357, 1.0890157, 1.0798485, 1.117274,
    1.0493417, 1.0851887, 1.1038903, 1.0890896, 1.0457299, 1.0506674])

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)

print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")

MODEL_PATH = "models/gru_model.keras"
custom_objects = {'mse': mse}
model = tf.keras.models.load_model(MODEL_PATH , custom_objects=custom_objects)
print(model.input_shape)  # Shape of input data
print(model.output_shape)  # Shape of the model's output
