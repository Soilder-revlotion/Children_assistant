"""
简易 HTTP API 服务器 — 绕过 uvicorn + PyTorch segfault 问题
用法: python scripts/serve_api.py --port 8000
"""
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.rag_engine import RAGEngine


class APIHandler(BaseHTTPRequestHandler):
    engine = None  # 类级别共享

    @classmethod
    def get_engine(cls):
        if cls.engine is None:
            print("[INIT] Loading RAG engine...")
            cls.engine = RAGEngine()
            print(f"[INIT] RAG engine ready, chromadb={cls.engine.use_chromadb}, embedding={cls.engine.use_embedding}")
        return cls.engine

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
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

            import time
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

    server = HTTPServer(("0.0.0.0", args.port), APIHandler)
    print(f"育儿助手 API: http://0.0.0.0:{args.port}")
    print(f"Health check: http://localhost:{args.port}/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
