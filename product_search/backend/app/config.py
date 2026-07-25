"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "product_search/backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Product Recommendation API"
    database_url: str = Field(min_length=1, repr=False)
    openai_api_key: str = Field(default="", repr=False)
    query_model: str = "gpt-5.4-nano"
    recommendation_model: str = "gpt-5.4-nano"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: PositiveInt = 768
    openai_timeout_seconds: PositiveFloat = 30.0
    database_timeout_seconds: PositiveFloat = 5.0
    default_top_k: int = Field(default=3, ge=1, le=20)
    default_candidate_limit: PositiveInt = 200
    database_pool_min_size: int = Field(default=1, ge=0)
    database_pool_max_size: int = Field(default=5, ge=1)
    auth_username: str = "api-user"
    auth_password: str = Field(default="", repr=False)
    jwt_secret_key: str = Field(default="", repr=False)
    jwt_issuer: str = "ece1508-product-search"
    jwt_audience: str = "ece1508-product-search-api"
    jwt_access_token_minutes: int = Field(default=30, ge=1, le=1440)


@lru_cache
def get_settings() -> Settings:
    return Settings()
