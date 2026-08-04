"""Access JWT verification — same contract as pangeaMedia / pangeaConversations."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

ACCESS_TYP = "access"
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    user_id: int
    claims: Dict[str, Any]


def decode_access_token(token: str, secret: str) -> Dict[str, Any]:
    payload = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        options={"require": ["exp"], "verify_expiration": True},
    )
    if not isinstance(payload, dict):
        raise jwt.InvalidTokenError("Invalid payload")
    if payload.get("typ") != ACCESS_TYP:
        raise jwt.InvalidTokenError("Token typ must be access")
    if payload.get("user_id") is None:
        raise jwt.InvalidTokenError("Missing user_id")
    return payload


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> AuthUser:
    if not settings.jwt_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_SECRET_KEY is not configured",
        )
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthenticated",
        )
    try:
        payload = decode_access_token(credentials.credentials, settings.jwt_secret_key)
        user_id = int(payload["user_id"])
        if user_id <= 0:
            raise ValueError("invalid user_id")
    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthenticated",
        ) from None

    return AuthUser(user_id=user_id, claims=payload)
