"""
育儿助手 RAG 后端 API

启动: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from backend.config import HOST, PORT, USE_RERANKER, USE_HYDE, RERANKER_MODEL
from backend.rag_engine import get_rag_engine

app = FastAPI(
    title="育儿助手 API",
    description="基于 RAG 的育儿知识问答系统",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    """预加载 RAG 引擎，避免请求时才加载导致 Windows 下 PyTorch 线程冲突"""
    get_rag_engine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    age_range: Optional[str] = None   # 可选：按年龄段过滤
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    has_knowledge: bool
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "service": "育儿助手 RAG API"}


@app.post("/api/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    """育儿问答接口"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    engine = get_rag_engine()
    t0 = time.time()
    result = engine.query(req.question)
    latency = (time.time() - t0) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        has_knowledge=result["has_knowledge"],
        latency_ms=round(latency, 1),
    )


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """知识库统计"""
    engine = get_rag_engine()
    doc_count = 0
    if engine.use_chromadb and engine.collection:
        doc_count = engine.collection.count()
    elif engine.tfidf_docs is not None:
        doc_count = len(engine.tfidf_docs)

    return {
        "indexed_documents": doc_count,
        "index_type": "chromadb" if engine.use_chromadb else "tfidf",
        "llm_provider": engine.llm_config.get("model", "unknown"),
        "reranker": {"enabled": USE_RERANKER, "model": RERANKER_MODEL} if USE_RERANKER else {"enabled": False},
        "hyde": {"enabled": USE_HYDE},
    }


# 静态文件服务（前端界面）
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    # Mount frontend dir for JS/CSS assets if needed
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
