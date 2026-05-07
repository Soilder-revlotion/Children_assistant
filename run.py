"""
育儿助手 RAG 系统 - 一键启动脚本

用法:
  python run.py              # 启动 API 服务器
  python run.py --rebuild    # 重建索引后启动
  python run.py --index-only # 仅重建索引
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="育儿助手 RAG 系统")
    parser.add_argument("--rebuild", action="store_true", help="重建知识库索引")
    parser.add_argument("--index-only", action="store_true", help="仅重建索引，不启动服务器")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    args = parser.parse_args()

    # 确保在项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 加载 .env 文件
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        print(f"[INFO] Loading .env from {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not os.getenv(key):
                        os.environ[key] = value

    # 检查 API Key
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "vllm": None,  # 不需要 key
    }
    key_name = key_map.get(provider)
    if key_name and not os.getenv(key_name):
        print(f"[WARN] {key_name} not set. LLM generation will not work.")
        print(f"[WARN] Copy .env.example to .env and set your API key.")
        print(f"[WARN] RAG retrieval will still work for testing.\n")

    # 检查/重建索引
    index_path = os.path.join("data", "chromadb", "tfidf_index.npz")
    if args.rebuild or args.index_only or not os.path.exists(index_path):
        print("[INFO] Building index...")
        import subprocess
        subprocess.run([sys.executable, "scripts/build_index.py"], check=True)

    if args.index_only:
        print("[DONE] Index built. Exiting.")
        return

    # 预加载 RAG 引擎（BGE 模型 + ChromaDB），避免首次请求超时
    print("\n[INFO] Preloading RAG engine...")
    from backend.rag_engine import get_rag_engine
    engine = get_rag_engine()
    print(f"[INFO] RAG engine ready. Index: {engine.collection.count() if engine.collection else 0} docs")

    # 启动服务器
    print(f"\n[INFO] Starting server on {args.host}:{args.port}")
    print(f"[INFO] API docs: http://{args.host}:{args.port}/docs")
    print(f"[INFO] Health: http://{args.host}:{args.port}/health")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
