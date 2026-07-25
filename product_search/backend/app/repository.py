"""Postgres/pgvector product retrieval."""

from collections.abc import Sequence

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .exceptions import DatabaseUnavailableError
from .schemas import Product, SearchQuery


SORT_EXPRESSIONS = {
    "relevance": "cosine_distance ASC, reviews DESC",
    "price_low_to_high": "price ASC, cosine_distance ASC",
    "price_high_to_low": "price DESC, cosine_distance ASC",
    "rating": "stars DESC, reviews DESC, cosine_distance ASC",
    "popularity": "bought_in_last_month DESC, reviews DESC, cosine_distance ASC",
}


def to_pgvector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def build_search_statement(
    query: SearchQuery,
    *,
    embedding_model: str,
    query_vector: str,
    limit: int,
    candidate_limit: int,
) -> tuple[str, list[object]]:
    where_clauses = ["e.model_name = %s", "p.price != 0"]
    filter_values: list[object] = [embedding_model]

    filters = (
        ("price_min", "p.price >= %s"),
        ("price_max", "p.price <= %s"),
        ("min_stars", "p.stars >= %s"),
        ("min_reviews", "p.reviews >= %s"),
    )
    for field, clause in filters:
        value = getattr(query, field)
        if value is not None:
            where_clauses.append(clause)
            filter_values.append(value)
    if query.brand:
        where_clauses.append("p.title ILIKE %s")
        filter_values.append(f"%{query.brand}%")
    if query.main_category:
        where_clauses.append("p.main_category = %s")
        filter_values.append(query.main_category)

    order_sql = SORT_EXPRESSIONS[query.sort_by]
    where_sql = " AND ".join(where_clauses)
    statement = f"""
        WITH semantic_candidates AS (
            SELECT
                p.asin, p.title, p.img_url, p.product_url, p.price,
                p.list_price, p.stars, p.reviews, p.is_best_seller,
                p.bought_in_last_month, p.main_category,
                e.title_embedding <=> %s::vector AS cosine_distance
            FROM amazon_product_title_embeddings AS e
            JOIN amazon_products AS p ON p.asin = e.asin
            WHERE {where_sql}
            ORDER BY e.title_embedding <=> %s::vector
            LIMIT %s
        )
        SELECT
            asin, title, img_url, product_url, price, list_price, stars,
            reviews, is_best_seller, bought_in_last_month, main_category,
            1.0 - cosine_distance AS similarity
        FROM semantic_candidates
        ORDER BY {order_sql}
        LIMIT %s
    """
    params: list[object] = [
        query_vector,
        *filter_values,
        query_vector,
        candidate_limit,
        limit,
    ]
    return statement, params


class PgProductRepository:
    def __init__(
        self,
        *,
        database_url: str,
        embedding_model: str,
        timeout_seconds: float,
        min_size: int,
        max_size: int,
    ) -> None:
        self.embedding_model = embedding_model
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout_seconds,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self._pool.open()

    def search(
        self,
        query: SearchQuery,
        query_embedding: Sequence[float],
        *,
        limit: int,
        candidate_limit: int,
    ) -> list[Product]:
        statement, params = build_search_statement(
            query,
            embedding_model=self.embedding_model,
            query_vector=to_pgvector_literal(query_embedding),
            limit=limit,
            candidate_limit=candidate_limit,
        )
        try:
            with self._pool.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(statement, params)
                    rows = cursor.fetchall()
            return [Product.model_validate(row) for row in rows]
        except Exception as exc:
            raise DatabaseUnavailableError(
                "Product database could not complete the search"
            ) from exc

    def ping(self) -> bool:
        try:
            with self._pool.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone() is not None
        except Exception:
            return False

    def close(self) -> None:
        self._pool.close()
