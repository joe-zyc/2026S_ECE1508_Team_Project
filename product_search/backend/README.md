# Product Recommendation Backend

This FastAPI service turns a natural-language shopping request into structured
filters, retrieves products through PostgreSQL/pgvector, and generates grounded
recommendation reasons.

## Setup

Run from the repository root so the `product_search` package is importable:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r product_search/backend/requirements.txt
cp product_search/backend/.env.example product_search/backend/.env
```

Edit `product_search/backend/.env` and configure all required values:

```env
DATABASE_URL=postgresql://amazon_user:amazon_password@localhost:5432/amazon_products
OPENAI_API_KEY=your-openai-api-key
AUTH_USERNAME=api-user
AUTH_PASSWORD=your-strong-password
JWT_SECRET_KEY=your-random-secret-of-at-least-32-bytes
```

`DATABASE_URL` has no application default. The backend will fail configuration
validation if it is missing or empty. For local execution, it must be a
connection URL reachable from the host Python process. The database must
contain the tables created by `data_prep/sql/schema.sql`, and its stored
embedding model must match `EMBEDDING_MODEL`.

Generate a suitable random JWT signing key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The remaining model, timeout, pool, JWT issuer/audience, and retrieval settings
have documented defaults in `.env.example` and can be changed there.

Start the API:

```bash
uvicorn product_search.backend.app.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Docker

The Compose service runs only the API and connects to the PostgreSQL/pgvector
database running on the host. Start the database from `data_prep` first, then
build and run the backend:

```bash
docker compose -f data_prep/docker-compose.yml up -d
cp product_search/backend/.env.example product_search/backend/.env
# Set OPENAI_API_KEY, AUTH_PASSWORD, and JWT_SECRET_KEY in the backend .env.
docker compose -f product_search/backend/docker-compose.yml up -d --build
```

Compose uses `BACKEND_DATABASE_URL` because `localhost` inside the API
container would refer to the container itself. Compose passes this value into
the container as its required `DATABASE_URL`. The example defaults to
`host.docker.internal`; `extra_hosts` enables the same name on Linux Docker.
Set `BACKEND_DATABASE_URL` in `product_search/backend/.env` when PostgreSQL is
hosted elsewhere:

```bash
BACKEND_DATABASE_URL=postgresql://user:password@database-host:5432/database
```

Then restart the service with
`docker compose -f product_search/backend/docker-compose.yml up -d`.

Check container state and logs:

```bash
docker compose -f product_search/backend/docker-compose.yml ps
docker compose -f product_search/backend/docker-compose.yml logs -f backend
```

The first startup can take several minutes while the embedding model is
downloaded. A named volume preserves the Hugging Face cache across container
replacements. The image uses CPU-only PyTorch and one worker to avoid loading
multiple copies of the embedding model. The image copies only the production
Python package; `query_constructor_pipeline.ipynb`, notebook checkpoints,
tests, and frontend files are not included.

Build or run the image without Compose from the repository root:

```bash
docker build -f product_search/backend/Dockerfile -t ece1508-product-search-backend .
docker run --rm -p 8000:8000 \
  --env-file product_search/backend/.env \
  --add-host host.docker.internal:host-gateway \
  -e DATABASE_URL=postgresql://amazon_user:amazon_password@host.docker.internal:5432/amazon_products \
  ece1508-product-search-backend
```

## API

Check readiness:

```bash
curl -i http://localhost:8000/health
```

Request recommendations:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"api-user","password":"your-password"}' \
  | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/v1/recommendations \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "query": "wireless gaming mouse under $50 with good ratings",
    "top_k": 3,
    "candidate_limit": 200
  }'
```

Strict filters are never relaxed. A valid search with no matching products
returns HTTP 200, an empty `recommendations` list, and an explanatory message.
Tokens are short-lived HS256 JWTs containing validated issuer, audience,
subject, issue-time, not-before, expiration, and unique-token claims. The
recommendations endpoint returns HTTP 401 for a missing, expired, or invalid
token. Health checks and interactive API documentation remain public.

## Architecture

- `query_constructor.py`: natural language to `SearchQuery`
- `embeddings.py`: replaceable query embedding provider
- `repository.py`: parameterized filtering and pgvector retrieval
- `recommendation_generator.py`: grounded, order-preserving reasons
- `service.py`: end-to-end orchestration
- `main.py`: lifecycle, health checks, HTTP routes, and error mapping

Components depend on the protocols in `interfaces.py`, allowing tests or future
fine-tuned models to replace individual implementations.

## Tests

```bash
pip install -r product_search/backend/requirements-dev.txt
pytest product_search/backend/tests
```

To enable the optional database integration test:

```bash
TEST_DATABASE_URL=postgresql://... pytest product_search/backend/tests -m integration
```
