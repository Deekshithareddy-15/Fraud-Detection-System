
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY api/ ./api/
COPY models/ ./models/
# Copy src just in case needed (though api seems standalone)
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run commands
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
