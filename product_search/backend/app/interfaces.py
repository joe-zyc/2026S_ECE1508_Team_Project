"""Replaceable pipeline component contracts."""

from typing import Protocol, Sequence

from .schemas import Product, SearchQuery


class QueryConstructor(Protocol):
    def construct(self, user_input: str) -> SearchQuery: ...


class EmbeddingProvider(Protocol):
    @property
    def ready(self) -> bool: ...

    def load(self) -> None: ...

    def embed(self, text: str) -> list[float]: ...


class ProductRepository(Protocol):
    def search(
        self,
        query: SearchQuery,
        query_embedding: Sequence[float],
        *,
        limit: int,
        candidate_limit: int,
    ) -> list[Product]: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class RecommendationGenerator(Protocol):
    def generate(
        self,
        user_request: str,
        parsed_query: SearchQuery,
        products: list[Product],
    ) -> list[str]: ...
