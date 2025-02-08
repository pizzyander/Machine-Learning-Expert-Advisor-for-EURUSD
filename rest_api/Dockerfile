# Use a base image that already has key ML libraries
FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY main.py .  
# Copy requirements.txt and install dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scikit-learn joblib

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
