"""Redis-backed session state for stateful multi-turn conversations."""
import json
import os
import redis.asyncio as aioredis

REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL   = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour
MAX_HISTORY   = 20  # max messages stored per session

_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _pool


async def get_history(session_id: str) -> list[dict]:
    """Return stored messages for this session (oldest first)."""
    r = get_redis()
    raw = await r.get(f"session:{session_id}")
    if raw is None:
        return []
    return json.loads(raw)


async def append_messages(session_id: str, messages: list[dict]) -> list[dict]:
    """Append new messages to session history, trim to MAX_HISTORY, reset TTL."""
    r       = get_redis()
    history = await get_history(session_id)
    history.extend(messages)
    history = history[-MAX_HISTORY:]
    await r.setex(f"session:{session_id}", SESSION_TTL, json.dumps(history))
    return history


async def delete_session(session_id: str) -> None:
    r = get_redis()
    await r.delete(f"session:{session_id}")
