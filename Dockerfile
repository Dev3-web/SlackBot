FROM python:3.10-slim

# Install OS dependencies for Playwright
RUN apt-get update && apt-get install -y 

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3000

# Run FastAPI with uvicorn
CMD ["python", "app.py"]
