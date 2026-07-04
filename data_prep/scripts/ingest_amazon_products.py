#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_DATABASE_URL = "postgresql://amazon_user:amazon_password@localhost:5432/amazon_products"
DEFAULT_CSV_PATH = "amazon_products_with_main_category.csv"
DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "sql" / "schema.sql"


PRODUCT_COLUMNS = (
    "asin",
    "title",
    "img_url",
    "product_url",
    "stars",
    "reviews",
    "price",
    "list_price",
    "category_id",
    "is_best_seller",
    "bought_in_last_month",
    "main_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Amazon product CSV rows and BGE title embeddings into Postgres/pgvector."
    )
    parser.add_argument(
        "--csv",
        default=os.getenv("CSV_PATH", DEFAULT_CSV_PATH),
        help="Path to amazon_products_with_main_category.csv.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="Postgres connection URL.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, for example cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N CSV/product rows. Useful for dry-runs.",
    )
    parser.add_argument(
        "--product-batch-size",
        type=int,
        default=10_000,
        help="Number of product rows per database upsert batch.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Number of titles per embedding model batch.",
    )
    parser.add_argument(
        "--skip-products",
        action="store_true",
        help="Do not load product rows; generate embeddings from rows already in the database.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Load products only; do not generate title embeddings.",
    )
    parser.add_argument(
        "--skip-vector-index",
        action="store_true",
        help="Do not create the pgvector HNSW index after embedding load.",
    )
    return parser.parse_args()


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, autocommit=False)


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def parse_product_row(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row["asin"],
        row["title"],
        row["imgUrl"],
        row["productURL"],
        float(row["stars"]),
        int(row["reviews"]),
        float(row["price"]),
        float(row["listPrice"]),
        int(row["category_id"]),
        row["isBestSeller"].strip().lower() == "true",
        int(row["boughtInLastMonth"]),
        row["main_category"],
    )


def limited_rows(csv_path: Path, limit: int | None) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def create_temp_products_table(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        CREATE TEMP TABLE tmp_amazon_products (
            asin TEXT,
            title TEXT,
            img_url TEXT,
            product_url TEXT,
            stars NUMERIC(3, 2),
            reviews INTEGER,
            price NUMERIC(12, 2),
            list_price NUMERIC(12, 2),
            category_id INTEGER,
            is_best_seller BOOLEAN,
            bought_in_last_month INTEGER,
            main_category TEXT
        );
        """
    )


def copy_rows(cur: psycopg.Cursor, table_name: str, columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    column_sql = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    query = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(table_name), column_sql)
    with cur.copy(query) as copy:
        for row in rows:
            copy.write_row(row)


def upsert_temp_products(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        INSERT INTO amazon_products (
            asin,
            title,
            img_url,
            product_url,
            stars,
            reviews,
            price,
            list_price,
            category_id,
            is_best_seller,
            bought_in_last_month,
            main_category
        )
        SELECT
            asin,
            title,
            img_url,
            product_url,
            stars,
            reviews,
            price,
            list_price,
            category_id,
            is_best_seller,
            bought_in_last_month,
            main_category
        FROM tmp_amazon_products
        ON CONFLICT (asin) DO UPDATE SET
            title = EXCLUDED.title,
            img_url = EXCLUDED.img_url,
            product_url = EXCLUDED.product_url,
            stars = EXCLUDED.stars,
            reviews = EXCLUDED.reviews,
            price = EXCLUDED.price,
            list_price = EXCLUDED.list_price,
            category_id = EXCLUDED.category_id,
            is_best_seller = EXCLUDED.is_best_seller,
            bought_in_last_month = EXCLUDED.bought_in_last_month,
            main_category = EXCLUDED.main_category,
            loaded_at = now();
        """
    )
    cur.execute("TRUNCATE tmp_amazon_products;")


def load_products(conn: psycopg.Connection, csv_path: Path, limit: int | None, batch_size: int) -> int:
    total = 0
    batch: list[tuple[Any, ...]] = []
    with conn.cursor() as cur:
        create_temp_products_table(cur)
        for row in tqdm(limited_rows(csv_path, limit), desc="Loading products", unit="rows"):
            batch.append(parse_product_row(row))
            if len(batch) >= batch_size:
                copy_rows(cur, "tmp_amazon_products", PRODUCT_COLUMNS, batch)
                upsert_temp_products(cur)
                conn.commit()
                total += len(batch)
                batch.clear()

        if batch:
            copy_rows(cur, "tmp_amazon_products", PRODUCT_COLUMNS, batch)
            upsert_temp_products(cur)
            conn.commit()
            total += len(batch)

    return total


def fetch_products_for_embedding(
    conn: psycopg.Connection, model_name: str, limit: int | None, batch_size: int
) -> Iterable[list[tuple[str, str]]]:
    seen = 0
    last_asin = ""

    while True:
        remaining = None if limit is None else limit - seen
        if remaining is not None and remaining <= 0:
            break

        current_batch_size = batch_size if remaining is None else min(batch_size, remaining)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.asin, p.title
                FROM amazon_products p
                LEFT JOIN amazon_product_title_embeddings e ON e.asin = p.asin
                WHERE p.asin > %s
                  AND (e.asin IS NULL OR e.model_name <> %s)
                ORDER BY p.asin
                LIMIT %s;
                """,
                (last_asin, model_name, current_batch_size),
            )
            rows = cur.fetchall()
            if not rows:
                break

        last_asin = rows[-1][0]
        seen += len(rows)
        yield [(asin, title) for asin, title in rows]


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def create_temp_embeddings_table(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        CREATE TEMP TABLE tmp_amazon_product_title_embeddings (
            asin TEXT,
            title_embedding vector(768),
            model_name TEXT
        );
        """
    )


def upsert_temp_embeddings(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        INSERT INTO amazon_product_title_embeddings (
            asin,
            title_embedding,
            model_name,
            embedding_created_at
        )
        SELECT asin, title_embedding, model_name, now()
        FROM tmp_amazon_product_title_embeddings
        ON CONFLICT (asin) DO UPDATE SET
            title_embedding = EXCLUDED.title_embedding,
            model_name = EXCLUDED.model_name,
            embedding_created_at = now();
        """
    )
    cur.execute("TRUNCATE tmp_amazon_product_title_embeddings;")


def load_embeddings(
    conn: psycopg.Connection,
    model_name: str,
    device: str | None,
    limit: int | None,
    batch_size: int,
) -> int:
    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    total = 0

    with conn.cursor() as cur:
        create_temp_embeddings_table(cur)
        for product_batch in tqdm(
            fetch_products_for_embedding(conn, model_name, limit, batch_size),
            desc="Embedding titles",
            unit="batch",
        ):
            asins = [asin for asin, _title in product_batch]
            titles = [title for _asin, title in product_batch]
            embeddings = model.encode(
                titles,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            actual_dimension = embeddings.shape[1]
            if actual_dimension != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Model {model_name} produced {actual_dimension}-dimensional embeddings, "
                    f"but the schema expects {EMBEDDING_DIMENSION}. Update sql/schema.sql "
                    "before using a different embedding model."
                )
            rows = [
                (asin, vector_literal(embedding), model_name)
                for asin, embedding in zip(asins, embeddings, strict=True)
            ]
            copy_rows(
                cur,
                "tmp_amazon_product_title_embeddings",
                ("asin", "title_embedding", "model_name"),
                rows,
            )
            upsert_temp_embeddings(cur)
            conn.commit()
            total += len(rows)

    return total


def create_vector_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_amazon_product_title_embeddings_hnsw
                ON amazon_product_title_embeddings
                USING hnsw (title_embedding vector_cosine_ops);
            """
        )
    conn.commit()


def print_summary(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM amazon_products;")
        product_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM amazon_product_title_embeddings;")
        embedding_count = cur.fetchone()[0]
        cur.execute(
            """
            SELECT vector_dims(title_embedding)
            FROM amazon_product_title_embeddings
            LIMIT 1;
            """
        )
        dimension_row = cur.fetchone()

    dimension = dimension_row[0] if dimension_row else None
    print(f"amazon_products rows: {product_count}")
    print(f"amazon_product_title_embeddings rows: {embedding_count}")
    print(f"sample embedding dimension: {dimension}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    csv_path = Path(args.csv)
    if not args.skip_products and not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with connect(args.database_url) as conn:
        ensure_schema(conn)

        if not args.skip_products:
            loaded = load_products(conn, csv_path, args.limit, args.product_batch_size)
            print(f"Loaded/upserted product rows: {loaded}")

        if not args.skip_embeddings:
            embedded = load_embeddings(
                conn,
                args.model_name,
                args.device,
                args.limit,
                args.embedding_batch_size,
            )
            print(f"Loaded/upserted embedding rows: {embedded}")

        if not args.skip_embeddings and not args.skip_vector_index:
            create_vector_index(conn)
            print("Ensured HNSW vector index exists.")

        print_summary(conn)


if __name__ == "__main__":
    main()
