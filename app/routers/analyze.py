"""POST /v1/analyze — core detection endpoint."""
import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.session import get_history, append_messages
from inference.predict import predict

router = APIRouter(prefix="/v1", tags=["analyze"])


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AnalyzeRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    session_id: str | None = None  # None = stateless, str = stateful


class AnalyzeResponse(BaseModel):
    session_id: str
    risk_score: float
    label: str
    attack_type: str | None
    attack_confidence: float | None
    flagged: bool
    latency_ms: float


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    _key: str = Depends(require_api_key),
):
    session_id = body.session_id or str(uuid.uuid4())
    new_msgs   = [m.model_dump() for m in body.messages]

    # Stateful mode: merge stored history with incoming messages
    history = await get_history(session_id)
    window  = history + new_msgs

    result = predict(window)

    # Persist the new messages (not the full window — history already stored)
    await append_messages(session_id, new_msgs)

    return AnalyzeResponse(
        session_id        = session_id,
        risk_score        = result["risk_score"],
        label             = result["label"],
        attack_type       = result.get("attack_type"),
        attack_confidence = result.get("attack_confidence"),
        flagged           = result["label"] != "safe",
        latency_ms        = result["latency_ms"],
    )


@router.delete("/session/{session_id}", status_code=204)
async def clear_session(
    session_id: str,
    _key: str = Depends(require_api_key),
):
    """Delete all stored history for a session."""
    from app.session import delete_session
    await delete_session(session_id)
