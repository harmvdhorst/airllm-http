import asyncio
from collections import OrderedDict
from typing import Any, Dict, Optional

from airllm import AirLLM


class ModelManager:
    """
    Multi-model manager with LRU caching.

    - AIRLLM_PRELOAD_MODELS: comma-separated model ids to preload
    - AIRLLM_MAX_MODELS: max number of models to keep in memory
    - Inference uses ONLY model.generate()
    """

    def __init__(self, default_model: str, max_models: int = 1):
        self._default_model = default_model
        self._max_models = max(1, int(max_models))

        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def _get_lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    async def preload(self, model_ids: list[str]) -> None:
        # Load sequentially to avoid memory spikes
        for mid in model_ids:
            if mid:
                await self.ensure_loaded(mid)

    async def ensure_loaded(self, model_id: Optional[str] = None):
        model_id = model_id or self._default_model

        # Fast path: cached
        async with self._cache_lock:
            if model_id in self._models:
                self._models.move_to_end(model_id, last=True)
                return self._models[model_id]

        # Per-model lock prevents duplicate loading of same model
        model_lock = self._get_lock(model_id)
        async with model_lock:
            async with self._cache_lock:
                if model_id in self._models:
                    self._models.move_to_end(model_id, last=True)
                    return self._models[model_id]

            model = await asyncio.to_thread(self._load_model_sync, model_id)

            async with self._cache_lock:
                self._models[model_id] = model
                self._models.move_to_end(model_id, last=True)
                await self._evict_if_needed(keep={model_id})

            return model

    def _load_model_sync(self, model_id: str):
        # Keep constructor flexible for different airllm versions
        if hasattr(AirLLM, "from_pretrained"):
            return AirLLM.from_pretrained(model_id)
        return AirLLM(model_id)

    async def _evict_if_needed(self, keep: set[str]):
        # Evict LRU models if cache exceeds cap
        while len(self._models) > self._max_models:
            lru_id, _ = next(iter(self._models.items()))
            if lru_id in keep:
                self._models.move_to_end(lru_id, last=True)
                continue
            self._models.pop(lru_id, None)

    async def unload_all(self):
        async with self._cache_lock:
            self._models.clear()

    async def generate(self, model_id: Optional[str], prompt: str, **kwargs: Any) -> str:
        model = await self.ensure_loaded(model_id)

        def _run() -> str:
            out = model.generate(prompt, **kwargs)
            if isinstance(out, str):
                return out
            if isinstance(out, dict) and "text" in out:
                return out["text"]
            if isinstance(out, (list, tuple)) and out and isinstance(out[0], str):
                return out[0]
            return str(out)

        return await asyncio.to_thread(_run)