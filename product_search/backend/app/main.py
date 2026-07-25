"""FastAPI entry point and application lifecycle."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Response, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import AuthPrincipal, AuthenticationError, JWTAuthService
from .config import Settings, get_settings
from .embeddings import SentenceTransformerEmbeddingProvider
from .exceptions import (
    DatabaseUnavailableError,
    EmbeddingUnavailableError,
    UpstreamModelError,
)
from .interfaces import EmbeddingProvider, ProductRepository
from .query_constructor import OpenAIQueryConstructor
from .recommendation_generator import OpenAIRecommendationGenerator
from .repository import PgProductRepository
from .schemas import (
    ErrorResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    TokenRequest,
    TokenResponse,
)
from .service import RecommendationService

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    service: RecommendationService | None = None
    repository: ProductRepository | None = None
    embedding_provider: EmbeddingProvider | None = None
    query_model_configured: bool = False
    auth_service: JWTAuthService | None = None


def build_runtime(settings: Settings) -> Runtime:
    runtime = Runtime(query_model_configured=bool(settings.openai_api_key))
    try:
        runtime.auth_service = JWTAuthService(
            username=settings.auth_username,
            password=settings.auth_password,
            secret_key=settings.jwt_secret_key,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            access_token_minutes=settings.jwt_access_token_minutes,
        )
    except ValueError:
        logger.exception("JWT authentication is not configured")

    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model, settings.embedding_dimension
    )
    runtime.embedding_provider = embedding_provider

    try:
        repository = PgProductRepository(
            database_url=settings.database_url,
            embedding_model=settings.embedding_model,
            timeout_seconds=settings.database_timeout_seconds,
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
        )
        runtime.repository = repository
    except Exception:
        logger.exception("Product repository initialization failed")
        return runtime

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not configured")
        return runtime

    query_constructor = OpenAIQueryConstructor(
        api_key=settings.openai_api_key,
        model=settings.query_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    recommendation_generator = OpenAIRecommendationGenerator(
        api_key=settings.openai_api_key,
        model=settings.recommendation_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    runtime.service = RecommendationService(
        query_constructor=query_constructor,
        embedding_provider=embedding_provider,
        product_repository=repository,
        recommendation_generator=recommendation_generator,
        default_candidate_limit=settings.default_candidate_limit,
    )
    try:
        embedding_provider.load()
    except EmbeddingUnavailableError:
        logger.exception("Embedding model initialization failed")
    return runtime


def create_app(
    *,
    settings: Settings | None = None,
    runtime: Runtime | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    supplied_runtime = runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = supplied_runtime or await run_in_threadpool(
            build_runtime, app_settings
        )
        yield
        active_runtime: Runtime = application.state.runtime
        if active_runtime.repository is not None:
            await run_in_threadpool(active_runtime.repository.close)

    app = FastAPI(
        title=app_settings.app_name,
        version="1.0.0",
        description="Natural-language product retrieval and grounded recommendations.",
        lifespan=lifespan,
    )
    bearer_scheme = HTTPBearer(auto_error=False)

    async def require_access_token(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    ) -> AuthPrincipal:
        current: Runtime = app.state.runtime
        if current.auth_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is not configured",
            )
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer access token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return current.auth_service.verify(credentials.credentials)
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    @app.exception_handler(UpstreamModelError)
    async def handle_upstream_error(_request, exc: UpstreamModelError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(DatabaseUnavailableError)
    async def handle_database_error(_request, exc: DatabaseUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(EmbeddingUnavailableError)
    async def handle_embedding_error(_request, exc: EmbeddingUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["operations"],
    )
    async def health(response: Response) -> HealthResponse:
        current: Runtime = app.state.runtime
        database_ready = bool(
            current.repository
            and await run_in_threadpool(current.repository.ping)
        )
        embedding_ready = bool(
            current.embedding_provider and current.embedding_provider.ready
        )
        ready = (
            database_ready
            and embedding_ready
            and current.query_model_configured
            and current.service is not None
            and current.auth_service is not None
        )
        if not ready:
            response.status_code = 503
        return HealthResponse(
            status="ok" if ready else "unavailable",
            database="ready" if database_ready else "unavailable",
            embedding_model="ready" if embedding_ready else "unavailable",
            query_model=(
                "configured"
                if current.query_model_configured
                else "unavailable"
            ),
            authentication=(
                "configured"
                if current.auth_service is not None
                else "unavailable"
            ),
        )

    @app.post(
        "/api/v1/auth/token",
        response_model=TokenResponse,
        responses={
            401: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["authentication"],
    )
    async def issue_token(request: TokenRequest) -> TokenResponse:
        current: Runtime = app.state.runtime
        if current.auth_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is not configured",
            )
        try:
            return current.auth_service.authenticate_and_issue(
                request.username, request.password
            )
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    @app.post(
        "/api/v1/recommendations",
        response_model=RecommendationResponse,
        responses={
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["recommendations"],
    )
    async def recommendations(
        request: RecommendationRequest,
        _principal: AuthPrincipal = Depends(require_access_token),
    ) -> RecommendationResponse:
        current: Runtime = app.state.runtime
        if current.service is None:
            raise EmbeddingUnavailableError(
                "Recommendation pipeline is not ready"
            )
        return await run_in_threadpool(
            current.service.recommend,
            request.query,
            top_k=request.top_k,
            candidate_limit=request.candidate_limit,
        )

    return app


app = create_app()
