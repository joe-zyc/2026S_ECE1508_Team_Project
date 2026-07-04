# Amazon Products Postgres + Title Embeddings

This project loads `amazon_products_with_main_category.csv` into Postgres and stores a separate pgvector table with embeddings of the product `title` column.

The default embedding model is `BAAI/bge-base-en-v1.5`, which produces 768-dimensional normalized vectors.

## Files

- `docker-compose.yml`: local Postgres database with pgvector.
- `sql/schema.sql`: product and title embedding tables.
- `scripts/ingest_amazon_products.py`: CSV loader and embedding pipeline.
- `requirements.txt`: Python dependencies.
- `.env.example`: default local environment variables.

## Setup

If you are using WSL and Docker is installed through Docker Desktop, enable WSL integration for this distro in Docker Desktop first.

Start Postgres:

```bash
docker compose up -d
```

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

## Dry Run

Load and embed the first 1,000 products:

```bash
bash scripts/run_ingestion.sh --limit 1000
```

Re-run the same command to verify idempotency. The row counts should stay at 1,000 instead of doubling.

## Full Load

Run the full CSV load and embedding job:

```bash
bash scripts/run_ingestion.sh
```

The CSV has 1,426,337 product rows. Full embedding generation can take a long time on CPU. Use `--device cuda` if you have a compatible GPU:

```bash
bash scripts/run_ingestion.sh --device cuda
```

## Useful Commands

Load products only:

```bash
bash scripts/run_ingestion.sh --skip-embeddings
```

Generate embeddings for already-loaded products:

```bash
bash scripts/run_ingestion.sh --skip-products
```

Skip HNSW index creation during experimentation:

```bash
bash scripts/run_ingestion.sh --limit 1000 --skip-vector-index
```

## Verification Queries

Connect to Postgres:

```bash
psql postgresql://amazon_user:amazon_password@localhost:5432/amazon_products
```

Check row counts:

```sql
SELECT count(*) FROM amazon_products;
SELECT count(*) FROM amazon_product_title_embeddings;
SELECT vector_dims(title_embedding) FROM amazon_product_title_embeddings LIMIT 1;
```

Run a simple similarity query by comparing one product title embedding against the rest:

```sql
WITH query_product AS (
    SELECT asin, title_embedding
    FROM amazon_product_title_embeddings
    WHERE asin = 'B014TMV5YE'
)
SELECT p.asin,
       p.title,
       e.title_embedding <=> q.title_embedding AS cosine_distance
FROM query_product q
JOIN amazon_product_title_embeddings e ON e.asin <> q.asin
JOIN amazon_products p ON p.asin = e.asin
ORDER BY e.title_embedding <=> q.title_embedding
LIMIT 10;
```

## Share the Database State

After loading products and embeddings, create a portable Postgres dump:

```bash
bash scripts/export_db.sh
```

Give collaborators `backups/amazon_products_latest.dump` plus the project files. They can restore it with:

```bash
bash scripts/restore_db.sh backups/amazon_products_latest.dump
```

See [docs/sharing-database-state.md](docs/sharing-database-state.md) for the full process.
