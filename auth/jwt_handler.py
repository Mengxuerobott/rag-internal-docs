"""
auth/jwt_handler.py
───────────────────
JWT creation, verification, and a FastAPI dependency that extracts the
current user from an Authorization: Bearer <token> header.

In production you would validate JWTs against your IdP (Okta, Auth0, Azure AD).
This module ships with a tiny in-memory user store so the project works
end-to-end without an external auth service — swap get_demo_token() for your
real login flow.

Usage
─────
1. Get a token (demo):
       POST /auth/token   {"username": "alice", "password": "secret"}

2. Use the token:
       POST /query   Authorization: Bearer <token>

3. The `get_current_user` FastAPI dependency reads the token, verifies the
   signature, and injects a `CurrentUser` into the route handler.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings

# ── Crypto helpers ─────────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── Demo user store ────────────────────────────────────────────────────────────
# In production: replace with DB lookup (users table) or IdP token introspection.
# Passwords are bcrypt-hashed. Plain-text equivalents are in .env.example comments.
#
# username → {hashed_password, role, full_name}
_DEMO_USERS: dict[str, dict] = {
    "alice":   {"password": hash_password("secret"), "role": "hr",          "full_name": "Alice Chen"},
    "bob":     {"password": hash_password("secret"), "role": "engineering",  "full_name": "Bob Kumar"},
    "carol":   {"password": hash_password("secret"), "role": "finance",      "full_name": "Carol Smith"},
    "dave":    {"password": hash_password("secret"), "role": "management",   "full_name": "Dave Park"},
    "eve":     {"password": hash_password("secret"), "role": "employee",     "full_name": "Eve Torres"},
    "frank":   {"password": hash_password("secret"), "role": "legal",        "full_name": "Frank Liu"},
    "admin":   {"password": hash_password("secret"), "role": "admin",        "full_name": "System Admin"},
}


# ── Pydantic models ────────────────────────────────────────────────────────────
class CurrentUser(BaseModel):
    username: str
    role: str
    full_name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int   # seconds


# ── JWT helpers ────────────────────────────────────────────────────────────────
def create_access_token(username: str, role: str, full_name: str) -> str:
    """
    Create a signed JWT containing username, role, and expiry.

    Claims:
        sub   — username (standard JWT subject)
        role  — single role string (e.g. "hr")
        name  — display name
        exp   — expiry timestamp
        iat   — issued-at timestamp
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)

    payload = {
        "sub":  username,
        "role": role,
        "name": full_name,
        "iat":  now,
        "exp":  expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT. Raises HTTPException 401 on any failure.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("sub") is None:
            raise ValueError("Missing subject")
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ─────────────────────────────────────────────────────────
# oauth2_scheme reads the token from the Authorization: Bearer header.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """
    FastAPI dependency — inject into any route that needs the authenticated user.

    Usage:
        @app.post("/query")
        async def query(req: QueryRequest, user: CurrentUser = Depends(get_current_user)):
            ...
    """
    payload = decode_token(token)
    return CurrentUser(
        username=payload["sub"],
        role=payload["role"],
        full_name=payload.get("name", ""),
    )


def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[CurrentUser]:
    """
    Like get_current_user but returns None instead of raising 401.
    Useful for endpoints that work for both authenticated and anonymous users
    but return different content.
    """
    if not token:
        return None
    try:
        return get_current_user(token)
    except HTTPException:
        return None


# ── Login helper (used by /auth/token endpoint in api/main.py) ────────────────
def authenticate_user(username: str, password: str) -> Optional[CurrentUser]:
    """
    Validate credentials against the demo user store.
    Returns CurrentUser on success, None on failure.
    """
    user = _DEMO_USERS.get(username.lower())
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return CurrentUser(username=username.lower(), role=user["role"], full_name=user["full_name"])


def login(form: OAuth2PasswordRequestForm) -> TokenResponse:
    """
    Authenticate and return a signed JWT.
    Called by the POST /auth/token route.
    """
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user.username, user.role, user.full_name)
    return TokenResponse(
        access_token=token,
        role=user.role,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )
