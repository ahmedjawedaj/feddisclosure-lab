FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY services/ ./services/
RUN pip install --no-cache-dir .
RUN useradd -m -u 10001 fdl && mkdir -p /data && chown -R fdl:fdl /data
USER fdl
EXPOSE 8000
CMD ["uvicorn", "services.client.app:app", "--host", "0.0.0.0", "--port", "8000"]
