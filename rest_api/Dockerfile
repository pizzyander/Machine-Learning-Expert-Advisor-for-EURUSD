FROM python:3.10

WORKDIR /app

COPY main.py .  
COPY requirements.txt .  

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
