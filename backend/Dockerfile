FROM python:3.14-slim AS builder
WORKDIR /build
COPY backend/pyproject.toml ./
COPY backend/app ./app
RUN python -m pip install --no-cache-dir --prefix=/install '.[embeddings]'

FROM python:3.14-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 appuser && \
    mkdir -p /data/documents /data/models && chown -R appuser:appuser /data
COPY --from=builder /install /usr/local
WORKDIR /app/backend
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
COPY backend/app ./app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
