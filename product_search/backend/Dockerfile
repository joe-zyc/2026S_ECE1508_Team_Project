# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Install the CPU build explicitly. This avoids pulling CUDA libraries into
# an API image that performs inference on CPU by default.
COPY product_search/backend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch && \
    python -m pip install -r /tmp/requirements.txt

# Create the runtime account before selecting it. The numeric USER declaration
# also lets Docker start the container without performing a username lookup.
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home appuser && \
    mkdir -p "${HF_HOME}"

# Copy only the importable production package. The exploratory notebook,
# checkpoints, tests, frontend, and local environment files are not image
# inputs.
COPY --chown=10001:10001 product_search/__init__.py /app/product_search/__init__.py
COPY --chown=10001:10001 product_search/backend/app /app/product_search/backend/app

RUN chown -R 10001:10001 "${HF_HOME}"

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "-m", "uvicorn", "product_search.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
