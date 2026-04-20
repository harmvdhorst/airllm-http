import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from model_manager import ModelManager

DEFAULT_MODEL = os.getenv("AIRLLM_MODEL", "garage-bAInd/Platypus2-70B-instruct")
MAX_MODELS = int(os.getenv("AIRLLM_MAX_MODELS", "2"))
API_KEY = (os.getenv("AIRLLM_API_KEY") or "").strip()


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


PRELOAD_MODELS = _parse_csv(os.getenv("AIRLLM_PRELOAD_MODELS", ""))

app = FastAPI(title="airllm-http", version="0.3.0")
manager = ModelManager(default_model=DEFAULT_MODEL, max_models=MAX_MODELS)


def _extract_bearer(auth_header: str) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def require_auth(request: Request):
    # If no key configured, auth disabled (handy for local dev).
    if not API_KEY:
        return

    auth = request.headers.get("authorization", "")
    bearer = _extract_bearer(auth)
    x_api_key = request.headers.get("x-api-key", "").strip()

    provided = bearer or x_api_key
    if not provided or provided != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

    # Any extra kwargs -> passed through to model.generate(...)
    extra: Dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
async def _startup():
    # Preload “array” of models at boot if configured
    if PRELOAD_MODELS:
        await manager.preload(PRELOAD_MODELS)


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.get("/ready", dependencies=[Depends(require_auth)])
async def ready():
    return {
        "ready": True,
        "default_model": DEFAULT_MODEL,
        "preload_models": PRELOAD_MODELS,
        "loaded_models": manager.loaded_models(),
        "max_models": MAX_MODELS,
        "auth_enabled": bool(API_KEY),
    }


@app.post("/unload", dependencies=[Depends(require_auth)])
async def unload():
    await manager.unload_all()
    return {"unloaded": True}


@app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
async def chat_completions(req: ChatCompletionsRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not implemented yet")

    model_id = req.model or DEFAULT_MODEL

    # Generic prompt formatting (works for most instruct/chat models)
    prompt = ""
    for m in req.messages:
        prompt += f"{m.role.upper()}: {m.content}\n"
    prompt += "ASSISTANT:"

    gen_kwargs: Dict[str, Any] = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.max_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_tokens
    if req.stop is not None:
        gen_kwargs["stop"] = req.stop

    gen_kwargs.update(req.extra or {})

    text = await manager.generate(model_id=model_id, prompt=prompt, **gen_kwargs)

    now = int(time.time())
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": now,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }
