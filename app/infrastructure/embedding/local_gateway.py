# -*- coding: utf-8 -*-
"""local_gateway —— 本地向量网关（OpenAI 兼容 /v1/embeddings）

## 为什么需要它

L5 基础设施里，检索链路有**两个**消费者都要求 OpenAI 兼容的 embedding 端点：

    商品向量索引   app/infrastructure/embedding/openai_embedding_client.py  → Qdrant
    品类知识库 RAG app/infrastructure/rag/category_knowledge.py            → AgentScope OpenAIEmbeddingModel

而本 Demo 接入的 LLM 网关（DeepSeek）只提供 chat/completions，`/v1/embeddings` 返回 404。
按分层架构，这属于「基础设施实现」的缺口，不该去改 application/domain 层，
所以在这里补一个本地进程，把**本地跑的小型中文向量模型**包装成 /v1/embeddings：

    LLM 网关       → app 直连（chat/completions）
    Embedding      → 本进程（127.0.0.1:8077/v1/embeddings）→ Qdrant

于是 application / domain / presentation 三层**一行都不用改**，
只需要 .env 里 `EMBEDDING_BASE_URL=http://127.0.0.1:8077/v1`。

## 两种后端

`EMBEDDING_BACKEND` 可显式指定，默认 auto（先试 fastembed，不可用则降级 hashing）：

    fastembed  本地 onnxruntime 跑 BAAI/bge-small-zh-v1.5（512 维），有语义泛化能力
    hashing    纯 Python 字符 n-gram 哈希向量（无第三方依赖），只能做字面相近匹配

hashing 后端存在的意义：即使模型下载失败，Demo 也能完整跑通五层链路
（召回策略会如实标注为向量召回，只是语义质量下降），而不是整条检索静默失效。

## 启动

    HF_ENDPOINT=https://hf-mirror.com \\
    uv run uvicorn app.infrastructure.embedding.local_gateway:app --port 8077

或直接用封装脚本：scripts/run_local_embedding.sh
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "bge-small-zh-v1.5")
_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()
_HASH_DIM = int(os.getenv("EMBEDDING_HASH_DIM", "512"))


# --------------------------------------------------------------------------- #
# 后端一：fastembed（本地 ONNX，有语义泛化能力）
# --------------------------------------------------------------------------- #

class _FastEmbedEncoder:
    """BAAI/bge-small-zh-v1.5（512 维）——中文短文本检索的性价比选择。

    模型文件从 HuggingFace 拉取，国内网络需要 HF_ENDPOINT 指向镜像，
    否则会卡在连接超时（本机实测 huggingface.co 不可达、hf-mirror.com 可用）。
    """

    name = "fastembed"

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # 延迟导入：装不上时走 hashing 兜底

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # fastembed 首次调用会触发 ONNX 图构建，先热身一次让首字延迟不在请求里
        probe = next(iter(self._model.embed(["热身"])))
        self.dim = len(probe)
        # onnxruntime 的 session 非线程安全，用锁串行化；embedding 是纯 CPU 短任务
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return [vector.tolist() for vector in self._model.embed(texts)]


# --------------------------------------------------------------------------- #
# 后端二：hashing（零依赖兜底）
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> list[str]:
    """中文 1-gram + 2-gram，英文按词并补 3-gram 前缀，兼顾中文无空格与英文词形。"""
    tokens: list[str] = []
    lowered = text.lower()
    for chunk in lowered.split():
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
            tokens.extend(chunk)
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
            if len(chunk) > 4:
                tokens.append(chunk[:4])
    return tokens


class _HashingEncoder:
    """字符 n-gram 哈希向量：确定性、零依赖，但只有字面相近性，无语义泛化。

    用 md5 而不是内置 hash()：后者带进程随机盐，换进程后同一段文本向量会变，
    向量库里的旧向量立刻失效（表现为"重建索引前后召回结果完全不同"）。
    """

    name = "hashing"

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # sublinear tf：抑制高频词的支配作用
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


def _build_encoder() -> Any:
    if _BACKEND == "hashing":
        logger.info("向量后端：hashing（按 EMBEDDING_BACKEND 显式指定，dim=%d）", _HASH_DIM)
        return _HashingEncoder()
    try:
        encoder = _FastEmbedEncoder(_DEFAULT_MODEL)
        logger.info("向量后端：fastembed（%s，dim=%d）", _DEFAULT_MODEL, encoder.dim)
        return encoder
    except Exception as err:  # noqa: BLE001 —— 兜底必须无条件生效
        logger.warning("fastembed 不可用（%s），降级 hashing 后端（无第三方依赖）", err)
        return _HashingEncoder()


# --------------------------------------------------------------------------- #
# HTTP 层：OpenAI /v1/embeddings 契约
# --------------------------------------------------------------------------- #

app = FastAPI(title="Globex 本地向量网关", version="1.0.0")

_encoder: Optional[Any] = None
_encoder_lock = threading.Lock()


def get_encoder() -> Any:
    """懒加载：uvicorn 启动后第一次请求才拉模型，避免启动阶段长时间无响应。"""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = _build_encoder()
    return _encoder


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: Optional[str] = None
    # 以下字段是 OpenAI 契约的一部分，收到即忽略：本地模型维度固定
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


@app.get("/health")
async def health() -> dict:
    encoder = get_encoder()
    return {"status": "ok", "backend": encoder.name, "dim": encoder.dim, "model": _DEFAULT_MODEL}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": _DEFAULT_MODEL, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingRequest) -> dict:
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input 不能为空")
    encoder = get_encoder()
    try:
        # 同步 CPU 计算，放线程池避免阻塞事件循环（跑批建库时动辄上百条）
        import anyio

        vectors = await anyio.to_thread.run_sync(encoder.encode, texts)
    except Exception as err:  # noqa: BLE001
        logger.exception("向量化失败")
        raise HTTPException(status_code=500, detail=f"向量化失败：{err}") from err
    return {
        "object": "list",
        "model": body.model or _DEFAULT_MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(text) for text in texts), "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("EMBEDDING_PORT", "8077")))
