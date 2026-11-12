# Basic demo container. Build: docker build -t predictive-maintenance-api .
# Run: docker run -p 5000:5000 predictive-maintenance-api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY models/ models/
COPY src/ src/
COPY examples/ examples/
EXPOSE 5000
CMD ["python", "src/app.py"]
