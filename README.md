# Children Assistant — 育儿助手 RAG 问答系统

基于 **RAG（检索增强生成）** 的育儿知识问答系统。整合默沙东诊疗手册、Wikipedia、合成蒸馏数据等来源，使用 BGE 语义检索 + 本地大模型生成专业育儿建议。

示例
<img width="1440" height="885" alt="image" src="https://github.com/user-attachments/assets/1b803bdd-97e8-4df6-9e01-bfdff7f76ce5" />

<img width="1440" height="885" alt="image" src="https://github.com/user-attachments/assets/5894f3e0-cb6a-44d4-871b-2ac8ed04ba38" />


## 系统架构

```
用户提问
    │
    ▼
┌──────────────────────────────────────┐
│  RAG 引擎                              │
│                                        │
│  ① BGE Embedding 语义检索 (ChromaDB)    │
│  ② 来源权重精排                         │
│  ③ LLM 生成 (Ollama / DeepSeek)        │
│                                        │
│  支持: HyDE 增强检索 / Reranker 精排     │
└──────────────────────────────────────┘
    │
    ▼
育儿建议 + 来源标注 + 免责声明
```

## 知识库

| 来源 | 条数 | 权重 | 说明 |
|------|:----:|:----:|------|
| MSD Manual（默沙东） | 161 | 1.0 | 专业医学手册，中文 |
| Wikipedia EN/ZH | 113 | 0.85 | 百科类权威来源 |
| R1 Distill（合成） | 3,173 | 0.5 | Qwen2.5 蒸馏生成 Q&A |

- **总计**: 3,447 条知识记录
- **向量索引**: ChromaDB 7,609 chunks，BGE-small-zh-v1.5 (512d)
- **覆盖**: 备孕 → 孕期 → 0-1月 → 1-6月 → 6-12月 → 1-3岁 → 3-6岁 → 6岁+

## 新电脑部署（8 步）

| 步骤 | 命令 | 说明 |
|:---:|------|------|
| 1 | 安装 Python 3.10+ | [python.org](https://www.python.org/downloads/) 下载，勾选"Add to PATH" |
| 2 | `git clone https://github.com/Soilder-revlotion/Children_assistant.git` | 拉取项目代码 |
| 3 | `pip install -r requirements.txt` | 安装 Python 依赖 |
| 4 | 安装 [Ollama](https://ollama.com) → `ollama pull qwen2.5:3b` | 下载本地模型（1.9GB） |
| 5 | `cp .env.example .env` | 创建配置文件，默认即用 |
| 6 | `python scripts/build_index.py` | **构建向量索引**（新电脑必须，约 3 分钟） |
| 7 | `python scripts/serve_api.py --port 8000` | 启动服务 |
| 8 | 浏览器打开 `http://localhost:8000` | 开始提问 |

> 详细说明（环境要求、验证测试、常见问题）→ [STARTUP.md](STARTUP.md)

## API 接口

### POST /api/chat

```json
// 请求
{ "question": "新生儿黄疸需要治疗吗", "top_k": 3 }

// 响应
{
  "answer": "新生儿黄疸分为生理性和病理性两种...",
  "sources": [
    {
      "title": "新生儿黄疸",
      "source": "msd_manual",
      "similarity": 0.76,
      "url": "https://www.msdmanuals.cn/...",
      "source_weight": 1.0
    }
  ],
  "has_knowledge": true,
  "latency_ms": 8234.5
}
```

### GET /api/knowledge/stats

```json
{
  "indexed_documents": 7609,
  "index_type": "chromadb",
  "llm_provider": "qwen2.5:3b"
}
```

### GET /health

```json
{ "status": "ok", "service": "育儿助手 RAG API" }
```

## 可选增强

`.env` 中按需开启：

| 功能 | 配置 | 说明 |
|------|------|------|
| HyDE | `USE_HYDE=true` | 假设回答增强检索召回 |
| Reranker | `USE_RERANKER=true` | BGE-Reranker 精排（需 1.1GB 显存） |
| DeepSeek API | `LLM_PROVIDER=deepseek` | 替代本地模型，需 API Key |

## 数据管道

```bash
# 爬取新数据源
python crawlers/crawl_msd_manual.py          # 默沙东诊疗手册

# 数据蒸馏（从现有知识库生成 Q&A）
python scripts/distill_data.py --samples 100 --qa-only

# 处理管道：合并 → 去重 → PII 脱敏 → 构建知识库
python scripts/process_knowledge.py

# 质量增强：实体识别 + 评分 + 缺口分析
python scripts/enhance_knowledge.py

# 重建向量索引
python scripts/build_index.py                 # 全量重建
python scripts/build_index.py --incremental   # 增量更新
```

## 项目结构

```
├── backend/
│   ├── config.py             # 配置管理（.env 加载）
│   ├── main.py               # FastAPI 应用
│   └── rag_engine.py         # RAG 引擎核心
├── crawlers/
│   ├── crawl_msd_manual.py   # MSD Manual Playwright 爬虫
│   └── crawl_wikipedia.py    # Wikipedia API 爬虫
├── scripts/
│   ├── serve_api.py          # 简易 HTTP Server（Windows 推荐）
│   ├── process_knowledge.py  # 数据处理管道
│   ├── build_index.py        # BGE/ChromaDB 索引构建
│   ├── enhance_knowledge.py  # 知识库质量增强
│   ├── distill_data.py       # R1 蒸馏数据生成
│   └── common_schema.py      # 数据模型
├── knowledge_base/           # 知识库文件（JSONL）
├── data/
│   ├── raw/                  # 原始爬取数据
│   ├── chromadb/             # 向量数据库（本地，不提交）
│   └── processed/            # 处理统计
├── .env.example              # 环境变量模板
├── STARTUP.md                # 详细部署指南
└── requirements.txt
```

## 技术栈

- **检索**: BGE-small-zh-v1.5 + ChromaDB（语义向量检索）
- **生成**: Ollama / Qwen2.5:3b（本地推理）
- **后端**: FastAPI + Python
- **爬虫**: Playwright（JS 渲染页面）+ Wikipedia API
- **回退**: TF-IDF（ChromaDB 不可用时）

## License

MIT
