FROM tensorflow/tensorflow:latest

WORKDIR /app 
 
# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training.py . 

# Install cron
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Add cron job to execute training script every Friday at 10:30 PM
RUN echo "30 22 * * 5 python /app/training.py" > /etc/cron.d/training_cron

# Give execution permission
RUN chmod 0644 /etc/cron.d/training_cron

# Apply crontab and start cron service
CMD ["sh", "-c", "cron && tail -f /dev/null"]
