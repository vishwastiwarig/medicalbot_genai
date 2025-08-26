# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Expose FastAPI port
EXPOSE 8000

# FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
