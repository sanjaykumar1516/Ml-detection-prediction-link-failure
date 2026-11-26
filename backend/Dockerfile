FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 120 -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 120 torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
