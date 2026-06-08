"""API key authentication via X-API-Key header."""
import os
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_header)) -> str:
    valid = os.getenv("JAILGUARD_API_KEY", "dev-secret-key")
    if not key or key != valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return key
