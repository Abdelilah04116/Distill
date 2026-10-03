# Image de service : expose /predict devant un modele deja exporte dans Ollama.
# Elle ne contient ni torch ni CUDA - l'inference se fait dans Ollama, pas ici.
FROM python:3.11-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY distill ./distill
RUN pip install --no-cache-dir --upgrade pip build \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".[serve]"

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Distill" \
      org.opencontainers.image.description="API locale devant un SLM specialise servi par Ollama." \
      org.opencontainers.image.authors="Abdelilah OURTI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/Abdelilah04116/distill"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    DISTILL_TASK=/app/task/task.yaml

WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels distill-slm[serve] \
 && rm -rf /wheels \
 && useradd --create-home --uid 10001 distill
COPY examples/email-triage /app/task

USER distill
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "distill.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
