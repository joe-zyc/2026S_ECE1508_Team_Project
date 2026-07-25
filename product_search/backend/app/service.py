"""End-to-end recommendation orchestration."""

from .interfaces import (
    EmbeddingProvider,
    ProductRepository,
    QueryConstructor,
    RecommendationGenerator,
)
from .schemas import (
    Recommendation,
    RecommendationResponse,
)


class RecommendationService:
    def __init__(
        self,
        *,
        query_constructor: QueryConstructor,
        embedding_provider: EmbeddingProvider,
        product_repository: ProductRepository,
        recommendation_generator: RecommendationGenerator,
        default_candidate_limit: int = 200,
    ) -> None:
        self._query_constructor = query_constructor
        self._embedding_provider = embedding_provider
        self._product_repository = product_repository
        self._recommendation_generator = recommendation_generator
        self._default_candidate_limit = default_candidate_limit

    def recommend(
        self,
        user_request: str,
        *,
        top_k: int,
        candidate_limit: int | None = None,
    ) -> RecommendationResponse:
        resolved_candidate_limit = candidate_limit or max(
            self._default_candidate_limit, top_k * 20
        )
        if resolved_candidate_limit < top_k:
            raise ValueError("candidate_limit must be greater than or equal to top_k")

        parsed_query = self._query_constructor.construct(user_request)
        embedding = self._embedding_provider.embed(
            parsed_query.product_description
        )
        products = self._product_repository.search(
            parsed_query,
            embedding,
            limit=top_k,
            candidate_limit=resolved_candidate_limit,
        )
        if not products:
            return RecommendationResponse(
                query=user_request,
                parsed_query=parsed_query,
                recommendations=[],
                message=(
                    "No products matched all requested constraints. "
                    "Try broadening the description or relaxing a filter."
                ),
            )

        reasons = self._recommendation_generator.generate(
            user_request, parsed_query, products
        )
        recommendations = [
            Recommendation(
                rank=rank,
                asin=product.asin,
                title=product.title,
                img_url=product.img_url,
                product_url=product.product_url,
                price=product.price,
                stars=product.stars,
                reviews=product.reviews,
                main_category=product.main_category,
                similarity=product.similarity,
                reason=reason,
            )
            for rank, (product, reason) in enumerate(
                zip(products, reasons, strict=True), start=1
            )
        ]
        return RecommendationResponse(
            query=user_request,
            parsed_query=parsed_query,
            recommendations=recommendations,
            message=f"Found {len(recommendations)} matching product recommendations.",
        )
