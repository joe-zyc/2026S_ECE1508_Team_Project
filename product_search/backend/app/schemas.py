"""Validated domain and API models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SortMode = Literal[
    "relevance",
    "price_low_to_high",
    "price_high_to_low",
    "rating",
    "popularity",
]


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_description: str = Field(min_length=1)
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    min_stars: float | None = Field(default=None, ge=0, le=5)
    min_reviews: int | None = Field(default=None, ge=0)
    brand: str | None = None
    main_category: str | None = None
    sort_by: SortMode = "relevance"

    @model_validator(mode="after")
    def validate_query(self) -> "SearchQuery":
        self.product_description = self.product_description.strip()
        if not self.product_description:
            raise ValueError("product_description cannot be empty")
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min cannot be greater than price_max")
        self.brand = self.brand.strip() if self.brand else None
        self.main_category = self.main_category.strip() if self.main_category else None
        return self


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "wireless gaming mouse under $50 with good ratings",
                    "top_k": 3,
                    "candidate_limit": 200,
                }
            ]
        },
    )

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    candidate_limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> "RecommendationRequest":
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query cannot be blank")
        if self.candidate_limit is not None and self.candidate_limit < self.top_k:
            raise ValueError("candidate_limit must be greater than or equal to top_k")
        return self


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asin: str
    title: str
    img_url: str
    product_url: str
    price: float
    list_price: float | None = None
    stars: float
    reviews: int
    is_best_seller: bool
    bought_in_last_month: int
    main_category: str
    similarity: float


class Recommendation(BaseModel):
    rank: int = Field(ge=1)
    asin: str
    title: str
    img_url: str
    product_url: str
    price: float
    stars: float
    reviews: int
    main_category: str
    similarity: float
    reason: str


class RecommendationResponse(BaseModel):
    query: str
    parsed_query: SearchQuery
    recommendations: list[Recommendation]
    message: str


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(gt=0)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ready", "unavailable"]
    embedding_model: Literal["ready", "unavailable"]
    query_model: Literal["configured", "unavailable"]
    authentication: Literal["configured", "unavailable"]


class ErrorResponse(BaseModel):
    detail: str
