"""RAG 后端配置"""

import os
from pathlib import Path

# 加载项目根目录 .env 文件
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
                if _key and _val and _key not in os.environ:
                    os.environ[_key] = _val

# --- LLM 配置 ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | deepseek | openai | qwen | vllm

LLM_CONFIGS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        "api_key": "ollama",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    },
    "vllm": {
        "base_url": os.getenv("VLLM_URL", "http://localhost:8000/v1"),
        "model": os.getenv("VLLM_MODEL", "qwen2.5-7b-instruct"),
        "api_key": "not-needed",
    },
}

# --- Embedding 配置 ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIM = 512

# --- Reranker 配置 ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_TOP_K = 20   # 粗排取 top 20 送入 reranker
RERANKER_FINAL_K = 5  # 精排后保留 top 5

# --- HyDE 配置 ---
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
HYDE_MAX_TOKENS = 100  # 假设回答长度（短回答用于检索增强）

# --- 数据金字塔来源权重 ---
# 权威医学源 > 专业科普 > Wikipedia > 社区 QA > 蒸馏数据
SOURCE_WEIGHTS = {
    "msd_manual": 1.0,     # 默沙东诊疗手册 - 专业医学手册
    "dxy": 1.0,           # 丁香园/丁香妈妈 - 专业医学
    "guideline": 1.0,      # 临床指南
    "who": 1.0,            # WHO
    "baike": 0.9,          # 百度百科
    "wikipedia_zh": 0.85,  # Wikipedia 中文
    "wikipedia_en": 0.8,   # Wikipedia 英文
    "zhihu": 0.7,          # 知乎高赞
    "babytree": 0.65,      # 宝宝树
    "mama_cn": 0.65,       # 妈妈网
    "xiaohongshu": 0.5,    # 小红书
    "r1_distill": 0.5,     # R1 蒸馏 - 通用模型生成，最弱
}

# --- 向量检索配置 ---
CHROMADB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.3

# --- 服务器配置 ---
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

# --- RAG Prompt 模板 ---
RAG_SYSTEM_PROMPT = """你是一个专业的育儿助手，基于提供的知识库内容回答用户问题。

请严格遵循以下规则：
1. 只根据【参考知识】中的内容回答问题，不要使用你自己的知识
2. 如果参考知识中没有相关信息，诚实地说"这个问题我目前的知识库中还没有覆盖到，建议咨询专业儿科医生"
3. 回答要准确、简洁、易懂，适合家长阅读
4. 如果涉及医疗建议，必须在回答末尾加上"以上内容仅供参考，不能替代专业医生的诊断和建议。如宝宝有不适，请及时就医。"
5. 回答时请标注引用的知识来源"""

RAG_USER_PROMPT = """【参考知识】
{context}

【用户问题】
{question}

请基于以上参考知识回答问题："""

# HyDE 假设回答 prompt（简短，仅用于辅助检索）
HYDE_PROMPT = """请用一段话（100字以内）简要回答以下育儿问题。不需要详细解释，只需要给出关键信息要点。

问题：{question}

简要回答："""

FALLBACK_RESPONSE = """抱歉，我目前的知识库中还没有覆盖到这个问题。

建议您：
- 咨询专业儿科医生或儿童保健专家
- 查阅权威育儿书籍或指南（如《美国儿科学会育儿百科》）
- 在丁香妈妈、育学园等专业平台搜索相关内容

我会持续学习和更新知识库，为您提供更好的服务。"""
