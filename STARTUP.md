# 育儿助手 RAG 系统 — 部署启动指南

## 环境要求

- Python 3.10+
- Ollama（本地 LLM，默认 http://localhost:11434）
- 已安装依赖: `pip install -r requirements.txt`

## 1. 前置检查

```bash
# 确认 Ollama 服务运行中
curl http://localhost:11434/api/tags

# 确认模型已拉取（默认 qwen2.5:3b）
ollama list

# 如需拉取模型
ollama pull qwen2.5:3b
```

## 2. 配置文件

编辑项目根目录 `.env`:
```bash
# --- LLM 本地模型 ---
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:3b

# --- 可选增强（按需开启）---
# USE_HYDE=true       # HyDE 假设回答增强
# USE_RERANKER=true   # Reranker 精排（需 1.1GB 显存）

# --- DeepSeek API（备选，需 API Key）---
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=sk-your-key-here
```

> `.env` 文件会自动加载，无需手动 source。

## 3. 启动 API 服务

```bash
cd D:\claude\1-1pachong

# 方式一：简易 HTTP Server（推荐，Windows 兼容）
python scripts/serve_api.py --port 8000

# 方式二：uvicorn（Linux/Mac，Windows 上可能出现 segfault）
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

启动后输出：
```
育儿助手 API: http://0.0.0.0:8000
Health check: http://localhost:8000/health
[INIT] Loading RAG engine...
[INFO] BGE embedding model loaded, dim=512
[INFO] ChromaDB connected, 7609 docs
[INIT] RAG engine ready, chromadb=True, embedding=True
```

## 4. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 知识库统计
curl http://localhost:8000/api/knowledge/stats

# 问答测试
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "新生儿母乳喂养有哪些注意事项", "top_k": 3}'
```

## 5. 数据管道（维护用）

```bash
# 爬取默沙东诊疗手册
python crawlers/crawl_msd_manual.py

# R1 蒸馏生成 Q&A 对
python scripts/distill_data.py --samples 100 --qa-only

# 数据处理：合并 → 去重 → PII 脱敏 → 构建知识库
python scripts/process_knowledge.py

# 知识库质量增强：标签 + 实体 + 评分 + 缺口分析
python scripts/enhance_knowledge.py

# 重建向量索引
python scripts/build_index.py              # 全量重建
python scripts/build_index.py --incremental # 增量更新
```

## 6. 项目目录结构

```
├── backend/
│   ├── config.py          # 配置 + .env 加载
│   ├── main.py            # FastAPI 应用
│   └── rag_engine.py      # RAG 引擎（检索 + 生成）
├── crawlers/
│   └── crawl_msd_manual.py # MSD Manual 爬虫
├── scripts/
│   ├── serve_api.py       # 简易 HTTP Server（推荐启动方式）
│   ├── process_knowledge.py # 数据处理管道
│   ├── build_index.py     # 向量索引构建
│   ├── enhance_knowledge.py # 知识库质量增强
│   ├── distill_data.py    # R1/Qwen 数据蒸馏
│   └── common_schema.py   # 公共数据模型
├── knowledge_base/
│   ├── parenting_knowledge_base.jsonl          # 主知识库 (3447条)
│   └── parenting_knowledge_base_enhanced.jsonl # 增强版（含实体/评分）
├── data/
│   ├── raw/               # 原始爬取数据
│   ├── chromadb/          # ChromaDB 向量索引
│   └── processed/         # 处理统计报告
└── .env                   # 环境变量配置
```

## 7. 常见问题

**Q: API 返回 "LLM 服务未配置"**
A: 检查 `.env` 文件是否存在，`LLM_PROVIDER` 是否正确，Ollama 是否运行。

**Q: Windows 上启动出现 segfault**
A: 使用 `python scripts/serve_api.py` 替代 uvicorn。

**Q: 检索结果来源全是 r1_distill**
A: 正常。知识库中 r1_distill 占 92%（3173/3447），MSD Manual 占 5%（161/3447）。专业医学查询会优先匹配 MSD Manual。

**Q: 如何升级 LLM**
A: 拉取更大模型 `ollama pull qwen2.5:7b`，修改 `.env` 中 `OLLAMA_MODEL=qwen2.5:7b`。
