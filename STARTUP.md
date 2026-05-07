# 育儿助手 RAG 系统 — 新电脑完整部署指南

从零开始在任意 Windows/Mac/Linux 电脑上部署本项目。

## 第一步：安装 Python

要求 Python **3.10 或以上**。

- 官网下载: https://www.python.org/downloads/
- 安装时勾选 "Add Python to PATH"

验证安装：
```bash
python --version   # 应显示 3.10+
```

## 第二步：克隆项目

```bash
git clone https://github.com/Soilder-revlotion/Children_assistant.git
cd Children_assistant
```

## 第三步：安装依赖

```bash
pip install -r requirements.txt
```

如果下载慢，用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 第四步：安装 Ollama 并下载模型

Ollama 是本地 AI 运行环境，项目用它来"思考"和生成回答。

1. 官网下载安装: https://ollama.com/download
2. 安装后自动在后台运行（任务栏有图标）
3. 打开终端/命令行，下载模型：

```bash
ollama pull qwen2.5:3b
```

> 模型约 1.9GB，下载需要几分钟。默认存在 C 盘，想改到其他盘：[Ollama 模型迁移指南](https://github.com/ollama/ollama#how-do-i-set-the-model-storage-location)

验证：
```bash
ollama list                    # 应该看到 qwen2.5:3b
curl http://localhost:11434/api/tags   # 应该返回 JSON
```

## 第五步：配置环境变量

```bash
# 复制配置模板
cp .env.example .env
```

默认配置即可使用（Ollama 本地模型），无需修改。`.env` 文件内容：

```bash
LLM_PROVIDER=ollama                  # 使用本地模型
OLLAMA_URL=http://localhost:11434/v1 # Ollama 地址
OLLAMA_MODEL=qwen2.5:3b             # 模型名称
PORT=8000                            # 端口号
```

## 第六步：构建向量索引（关键！）

> 索引文件（`data/chromadb/`）太大未上传 GitHub，新电脑必须自己构建一次。

```bash
python scripts/build_index.py
```

这个过程会：
- 加载 3447 条知识库记录
- 用 BGE 模型将每条知识转为 512 维语义向量
- 存入 ChromaDB

预计耗时 2-5 分钟（取决于 CPU）。

## 第七步：启动服务

```bash
python scripts/serve_api.py --port 8000
```

看到以下输出表示成功：
```
=======================================================
  育儿助手 RAG 系统启动中...
=======================================================
[1/2] 加载 BGE 嵌入模型 + ChromaDB 知识库...
      ChromaDB: 7609 条索引
      BGE 模型: 已加载
      LLM: qwen2.5:3b
      耗时: 8s
[2/2] 启动 HTTP 服务...
=======================================================
  Web 聊天界面: http://localhost:8000
=======================================================
  按 Ctrl+C 停止服务
```

## 第八步：打开浏览器

访问 **http://localhost:8000**

看到聊天界面，输入育儿问题即可。

---

## 验证是否正常

在另一个终端测试：

```bash
# 测试 1：健康检查
curl http://localhost:8000/health
# → {"status":"ok","service":"育儿助手 RAG API"}

# 测试 2：知识库统计
curl http://localhost:8000/api/knowledge/stats
# → {"indexed_documents":7609,"index_type":"chromadb","llm_provider":"qwen2.5:3b"}

# 测试 3：问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "新生儿母乳喂养有哪些注意事项"}'
# → 返回完整回答 + 参考来源
```

---

## 可选：使用云端 API 替代本地模型

如果电脑配置低（<8GB 内存），可以使用 DeepSeek 云端 API：

1. 注册 https://platform.deepseek.com 获取 API Key
2. 修改 `.env`：
   ```bash
   LLM_PROVIDER=deepseek
   DEEPSEEK_API_KEY=sk-你的key
   ```
3. 重启服务即可，无需 Ollama

---

## 常见问题

**Q: `pip install` 报错**
A: 尝试 `pip install --upgrade pip`，或使用国内镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q: `ollama pull` 下载很慢**
A: 设置代理或手动下载 GGUF 模型文件导入。参考: https://github.com/ollama/ollama#how-do-i-import-a-model

**Q: 启动时 segfault 崩溃（Windows 常见）**
A: 已解决。项目使用 `python scripts/serve_api.py` 而非 uvicorn 启动，避免了 PyTorch + uvicorn 的兼容问题。

**Q: 回答质量不好**
A: 默认模型 qwen2.5:3b 是 3B 小型模型，适合低配电脑。升级方式：
```bash
ollama pull qwen2.5:7b
# 修改 .env 中 OLLAMA_MODEL=qwen2.5:7b
```

**Q: 端口 8000 被占用**
A: `python scripts/serve_api.py --port 8080` 换个端口

**Q: 如何更新代码**
A: `git pull` 拉取最新，如果知识库有变化需重建索引：`python scripts/build_index.py`
