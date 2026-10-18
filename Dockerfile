FROM python:3.12-slim

ARG GID=1000
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && \
    useradd --no-create-home -u ${GID} gwop && \
    mkdir -p /app/data && chown gwop:gwop /app/data
USER gwop

EXPOSE 8011
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8011/health').read()" || exit 1

CMD ["python", "-m", "gemini_openai_proxy"]
