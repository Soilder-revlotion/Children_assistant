"""
育儿助手 HTTP API 服务器 — Windows 兼容版
用法: python scripts/serve_api.py --port 8000
"""
import json
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.rag_engine import RAGEngine

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


class APIHandler(BaseHTTPRequestHandler):
    engine = None

    @classmethod
    def get_engine(cls):
        return cls.engine  # 已在 main() 中预加载

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(os.path.join(FRONTEND_DIR, "index.html"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send_json({"status": "ok", "service": "育儿助手 RAG API"})
        elif self.path == "/api/knowledge/stats":
            engine = self.get_engine()
            doc_count = engine.collection.count() if engine.collection else 0
            self._send_json({
                "indexed_documents": doc_count,
                "index_type": "chromadb" if engine.use_chromadb else "tfidf",
                "llm_provider": engine.llm_config.get("model", "unknown"),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            question = req.get("question", "").strip()
            if not question:
                self._send_json({"error": "问题不能为空"}, 400)
                return

            engine = self.get_engine()
            t0 = time.time()
            result = engine.query(question)
            latency = (time.time() - t0) * 1000

            self._send_json({
                "answer": result["answer"],
                "sources": result["sources"],
                "has_knowledge": result["has_knowledge"],
                "latency_ms": round(latency, 1),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # ── 预加载 RAG 引擎（BGE 模型 + ChromaDB），避免首次请求卡住 ──
    print("=" * 55)
    print("  育儿助手 RAG 系统启动中...")
    print("=" * 55)
    print("[1/2] 加载 BGE 嵌入模型 + ChromaDB 知识库...")
    t0 = time.time()
    APIHandler.engine = RAGEngine()
    elapsed = time.time() - t0
    if APIHandler.engine.use_chromadb:
        print(f"      ChromaDB: {APIHandler.engine.collection.count()} 条索引")
    print(f"      BGE 模型: {'已加载' if APIHandler.engine.use_embedding else '未加载（回退 ChromaDB 内置）'}")
    print(f"      LLM: {APIHandler.engine.llm_config.get('model', '?')}")
    print(f"      耗时: {elapsed:.0f}s")

    # ── 启动服务器 ──
    print(f"\n[2/2] 启动 HTTP 服务...")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), APIHandler)
    print("=" * 55)
    print(f"  Web 界面: http://localhost:{args.port}")
    print(f"  API 文档: http://localhost:{args.port}/docs (仅 uvicorn 模式)")
    print(f"  健康检查: http://localhost:{args.port}/health")
    print("=" * 55)
    print("  按 Ctrl+C 停止服务\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
