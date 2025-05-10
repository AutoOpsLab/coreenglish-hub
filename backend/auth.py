# backend/auth.py
from datetime import datetime, timedelta
from select import select
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlmodel import Session
from fastapi import Request

from backend.db import User, engine

# secret – in production load from env
SECRET_KEY = "THIS_IS_ZAIN_GLENNS_SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": subject}
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


async def get_token_from_cookie_or_header(request: Request) -> str | None:
    # Check secure cookie first
    token = request.cookies.get("access_token")
    if token:
        return token

    # Fallback to Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()

    return None


async def get_current_user_optional(request: Request) -> User | None:
    token = await get_token_from_cookie_or_header(request)
    if not token:
        return None

    email = decode_access_token(token)
    if not email:
        return None

    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).one_or_none()
    return user
