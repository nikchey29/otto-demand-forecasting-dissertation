FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OTTO_ARTIFACT_DIR=/app/artifacts/research

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY artifacts/research ./artifacts/research

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "otto_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
