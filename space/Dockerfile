FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi==0.159.1 uvicorn==0.54.0 httpx==0.28.1
WORKDIR /app
COPY app.py .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
