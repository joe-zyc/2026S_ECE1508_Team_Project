"""JWT access-token issuance and validation."""

import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt

from .schemas import TokenResponse


class AuthenticationError(Exception):
    """Credentials or an access token could not be authenticated."""


@dataclass(frozen=True)
class AuthPrincipal:
    subject: str


class JWTAuthService:
    algorithm = "HS256"

    def __init__(
        self,
        *,
        username: str,
        password: str,
        secret_key: str,
        issuer: str,
        audience: str,
        access_token_minutes: int,
    ) -> None:
        if not username or not password:
            raise ValueError("JWT service credentials must be configured")
        if len(secret_key.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 bytes")
        self._username = username
        self._password = password
        self._secret_key = secret_key
        self._issuer = issuer
        self._audience = audience
        self._lifetime = timedelta(minutes=access_token_minutes)

    def authenticate_and_issue(self, username: str, password: str) -> TokenResponse:
        username_valid = hmac.compare_digest(username, self._username)
        password_valid = hmac.compare_digest(password, self._password)
        if not (username_valid and password_valid):
            raise AuthenticationError("Invalid username or password")

        now = datetime.now(timezone.utc)
        expires_at = now + self._lifetime
        claims = {
            "sub": f"service:{self._username}",
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "nbf": now,
            "exp": expires_at,
            "jti": str(uuid4()),
        }
        token = jwt.encode(
            claims,
            self._secret_key,
            algorithm=self.algorithm,
        )
        return TokenResponse(
            access_token=token,
            expires_in=int(self._lifetime.total_seconds()),
        )

    def verify(self, token: str) -> AuthPrincipal:
        try:
            claims = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self.algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": ["sub", "iss", "aud", "iat", "nbf", "exp", "jti"]
                },
            )
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError("Invalid or expired access token") from exc

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.startswith("service:"):
            raise AuthenticationError("Invalid or expired access token")
        return AuthPrincipal(subject=subject)
