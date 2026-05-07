# GitHub 高质量爬虫项目调研 & 推荐方案

> 目标：为育儿垂直大模型爬取高质量数据，用于知识蒸馏
> 日期：2026-05-05

---

## 一、GitHub 四大热门爬虫项目

| 项目 | Stars | 核心优势 |
|------|-------|---------|
| **[Crawl4AI](https://github.com/unclecode/crawl4ai)** | ~64K | **专为 LLM 设计**，直接输出 Markdown/JSON，内置 Playwright + 反爬，开箱即用 |
| **[Scrapy](https://github.com/scrapy/scrapy)** | 52.9K | 老牌异步框架，中间件生态丰富，可搭配 Playwright 插件 |
| **[Crawlee Python](https://github.com/apify/crawlee-python)** | 8K+ | **内置指纹伪装**、代理轮换、会话管理，与 Playwright 深度整合 |
| **[Scrapling](https://github.com/D4Vinci/Scrapling)** | 8.9K | **"不可检测"**，原生绕过 Cloudflare Turnstile，自适应 DOM 变化 |

### 各项目详情

#### 1. Crawl4AI —— LLM 数据管道首选

- 智能 Markdown 生成，自动去噪，BM25 算法提取核心内容
- LLM 驱动的结构化提取（支持 OpenAI/Claude/Gemini）
- 基于 Playwright，支持动态 JS 渲染、无限滚动、懒加载
- 隐身模式、代理轮换、User-Agent 轮换、反 Cloudflare 检测
- Docker 一键部署，内置 FastAPI + JWT 认证
- 安装：`pip install crawl4ai`

#### 2. Scrapy —— 成熟稳定的异步框架

- 代理中间件生态丰富
- 可通过 `scrapy-playwright` 插件处理 JS 渲染页面
- 适用于大规模分布式抓取
- 安装：`pip install scrapy`

#### 3. Crawlee Python —— 指纹伪装内置

- `PlaywrightCrawler` 默认开启浏览器指纹伪装
- 支持 Camoufox（隐身 Firefox fork）和 CloakBrowser（隐身 Chromium fork）
- 内置代理轮换、会话池、自动重试
- 支持 URL 队列持久化，断点续爬
- 安装：`pip install 'crawlee[playwright]'`

#### 4. Scrapling —— 反检测利器

- `StealthyFetcher` 自动解决 Cloudflare Turnstile
- TLS 指纹伪装（JA3/JA4），模拟 Chrome/Firefox/Safari
- 自适应元素定位，DOM 结构变化后自动重新跟踪
- 比 BeautifulSoup 快 620 倍
- 安装：`pip install scrapling`

---

## 二、反反爬 & 模拟真人方案

| 工具 | 用途 |
|------|------|
| **[playwright-stealth](https://github.com/Mattwmaster58/playwright_stealth)** | 隐藏 `navigator.webdriver` 等自动化痕迹 |
| **[humanization-playwright](https://github.com/saksham-personal/humanization-playwright)** | 贝塞尔曲线鼠标轨迹、变速键入、惯性滚动——全套真人模拟 |
| **[pydoll](https://github.com/autoscrape-labs/pydoll)** | 基于 CDP 协议，无需 WebDriver，原生过 CAPTCHA |
| **Playwright `storageState`** | **保存登录态**：登录一次，存为 JSON，后续复用，跳过每次登录 |

### Playwright storageState 用法

```python
# 保存登录态
await context.storage_state(path="auth.json")

# 复用登录态
context = await browser.new_context(storage_state="auth.json")
```

### 反检测技术栈（逐层叠加）

| 层级 | 工具 |
|------|------|
| 基础浏览器 | Patchright（打过补丁的 Playwright）或 Playwright + `playwright-stealth` |
| CDP 层 | `pystealth` 阻止 Runtime.enable 泄露 |
| 指纹层 | `playwright-stealth` 伪装模块，`browserforge` 生成真实指纹 |
| 行为层 | `humanization-playwright` 贝塞尔鼠标轨迹 + 随机延迟 |
| 网络层 | 轮换代理 IP + 轮换 User-Agent / Accept-Language |
| 框架层 | `Crawlee` PlaywrightCrawler 统筹以上所有 |

---

## 三、LLM 知识蒸馏数据管道范式

从 CCI3.0-HQ 和 ChineseWebText 等项目总结出的通用流水线：

```
CommonCrawl / 原始网页
        ↓
   爬虫抓取 + 语言识别
        ↓
   规则过滤（长度/汉字比例/敏感词）
        ↓
   大模型打分（Qwen2-72B / GPT-4 等）
        ↓
   知识蒸馏 → 轻量分类器（FastText / 0.5B BERT-like）
        ↓
   海量数据高效筛选 → 高质量预训练数据集
```

### 参考中文数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| [Chinese-DeepSeek-R1-Distill-data-110k](https://github.com/YunwenTechnology/Chinese-Data-Distill-From-R1) | 110K 条 | 含 reasoning_content + content + score |
| [ChineseWebText](https://github.com/CASIA-LM/ChineseWebText) | 1.42 TB | 含 600GB 高质量子集（质量分 > 90%） |
| [CCI3.0-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | 500GB | 智源 BAAI，Qwen2-72B 打分筛选 |
| [SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) | ~150B tokens | 中文预训练语料 |

---

## 四、中文育儿类数据源（建议爬取目标）

| 类型 | 站点举例 |
|------|---------|
| 母婴社区 | 宝宝树、妈妈网、育儿网 |
| 问答平台 | 知乎-育儿话题、百度知道-育儿 |
| 专业内容 | 丁香妈妈、育学园 |
| 电商评价 | 京东/天猫母婴商品评论 |
| 小红书 | 母婴笔记（需特殊处理 x-s 签名参数） |

---

## 五、推荐技术栈总览

```
┌─────────────────────────────────────┐
│          数据源（育儿网站）            │
├─────────────────────────────────────┤
│  Crawl4AI / Crawlee-Python  ← 调度层  │
│  Playwright + playwright-stealth  ← 浏览器自动化和反反爬  │
│  humanization-playwright  ← 真人行为模拟  │
│  storageState  ← 登录态持久化  │
│  BeautifulSoup / lxml  ← CSS 解析提取  │
├─────────────────────────────────────┤
│  输出：Markdown / JSONL（喂给蒸馏）    │
└─────────────────────────────────────┘
```

---

## 六、下一步行动建议

1. 首选 [Crawl4AI](https://github.com/unclecode/crawl4ai) —— 专门为 LLM 数据管道设计，省去大量格式转换工作
2. 搭配 [playwright-stealth](https://github.com/Mattwmaster58/playwright_stealth) 处理反爬
3. 用 Playwright 原生 `storageState` 保存各网站的登录态
4. 针对具体育儿站点设计 CSS 选择器提取正文内容
5. 参考 CCI3.0-HQ 的蒸馏筛选流程，构建高质量数据过滤管道
