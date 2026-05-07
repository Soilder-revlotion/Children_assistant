"""
DXY 丁香妈妈文章爬虫 — 两步走策略

Phase 1: httpx 并发扫描文章 ID → 提取标题 → 育儿关键词过滤
Phase 2: Playwright + auth cookies 提取正文 → KnowledgeItem 格式

用法:
  python crawlers/crawl_authenticated.py                  # 全量扫描
  python crawlers/crawl_authenticated.py --max 50          # 限制篇数
  python crawlers/crawl_authenticated.py --scan-only       # 仅扫描不爬取
  python crawlers/crawl_authenticated.py --start 200000    # 指定扫描起始 ID

设计要点:
  - 仅针对 DXY（知乎触发 CAPTCHA 验证，不可行）
  - httpx 异步并发 HEAD 检测 + GET 标题提取，速度快
  - Playwright 仅用于育儿相关文章的正文提取（DOM 渲染需浏览器）
  - 重度真人模拟：随机 UA/延迟/鼠标/滚动
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

AUTH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "auth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_articles.jsonl")
SCAN_CACHE_FILE = os.path.join(OUTPUT_DIR, "dxy_article_ids.json")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

PARENTING_KW = [
    "宝宝", "婴儿", "幼儿", "儿童", "新生儿", "孩子", "小孩",
    "育儿", "母婴", "亲子", "孕期", "怀孕", "孕妇", "产妇",
    "月子", "产后", "备孕", "胎教", "母乳", "配方奶", "奶粉",
    "辅食", "断奶", "喂奶", "发烧", "感冒", "咳嗽", "腹泻",
    "湿疹", "过敏", "黄疸", "疫苗", "发育", "睡眠", "早教",
    "儿科", "小儿", "婴幼儿", "学龄", "入园", "生长",
]

# 排除关键词：标题含这些的跳过（非育儿健康内容）
EXCLUDE_KW = [
    "做爱", "性爱", "性功能", "阳痿", "早泄", "勃起", "避孕",
    "痛风", "痔疮", "脱发", "秃头", "壮阳", "肾虚",
    "栏目收费", "商务合作", "广告", "招商",
]

# ─── 扫描配置 ───────────────────────────────────────────
SCAN_START = 180000      # 文章 ID 起始
SCAN_END = 212000        # 文章 ID 结束
SCAN_CONCURRENCY = 20    # httpx 并发数
SCAN_BATCH = 500         # 每批扫描 ID 数


# ═══════════════════════════════════════════════════════════
#  Phase 1: httpx 快速扫描
# ═══════════════════════════════════════════════════════════

def is_parenting_title(title: str) -> bool:
    """判断文章标题是否为育儿相关"""
    title_lower = title.lower()
    for kw in EXCLUDE_KW:
        if kw in title_lower:
            return False
    return any(kw in title for kw in PARENTING_KW)


async def check_article(client: httpx.AsyncClient, article_id: int,
                        semaphore: asyncio.Semaphore) -> dict | None:
    """检查单个文章 ID：返回 {id, title, url} 或 None"""
    url = f"https://dxy.com/article/{article_id}"

    async with semaphore:
        try:
            # HEAD 检查是否存在
            head_resp = await client.head(url, timeout=10)
            if head_resp.status_code != 200:
                return None

            # GET 提取标题（从 HTML <title> 标签）
            get_resp = await client.get(url, timeout=15,
                                        headers={"User-Agent": random.choice(USER_AGENTS)})
            if get_resp.status_code != 200:
                return None

            # 从 HTML 提取 title
            match = re.search(r"<title>(.+?)</title>", get_resp.text, re.IGNORECASE)
            if not match:
                return None

            title = match.group(1).strip()
            # DXY 标题格式: "标题内容|丁香医生" → 提取标题部分
            if "|" in title:
                title = title.split("|")[0].strip()

            if not title or len(title) < 4:
                return None

            if not is_parenting_title(title):
                return None

            return {"id": article_id, "title": title, "url": url}

        except Exception:
            return None


async def scan_article_ids(start: int, end: int, concurrency: int = SCAN_CONCURRENCY) -> list[dict]:
    """并发扫描文章 ID 范围，返回育儿相关文章列表"""
    print(f"[SCAN] Scanning article IDs {start}–{end} ({(end-start)} IDs, concurrency={concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                  limits=httpx.Limits(max_connections=concurrency)) as client:
        tasks = [check_article(client, aid, semaphore) for aid in range(start, end + 1)]
        results = []
        done = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            done += 1
            if result:
                results.append(result)
                print(f"  [{done}/{end-start+1}] FOUND: [{result['id']}] {result['title'][:60]}")
            if done % SCAN_BATCH == 0:
                print(f"  [SCAN PROGRESS] {done}/{end-start+1} checked, {len(results)} parenting articles found")

    print(f"\n[SCAN DONE] {len(results)} parenting articles out of {end-start+1} IDs scanned")
    return results


# ═══════════════════════════════════════════════════════════
#  Phase 2: Playwright 正文提取
# ═══════════════════════════════════════════════════════════

ANTI_DETECTION_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'chrome', { get: () => ({ runtime: {} }) });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
"""


async def human_delay(min_s=0.5, max_s=3.0):
    await asyncio.sleep(random.uniform(min_s, max_s))


async def simulate_mouse(page):
    for _ in range(random.randint(1, 3)):
        await page.mouse.move(
            random.randint(100, 800), random.randint(100, 600),
            steps=random.randint(3, 10)
        )
        await asyncio.sleep(random.uniform(0.05, 0.15))


async def simulate_scroll(page, scrolls=None):
    if scrolls is None:
        scrolls = random.randint(2, 5)
    for _ in range(scrolls):
        delta = random.randint(200, 500)
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(random.uniform(0.15, 0.5))
    if random.random() < 0.3:
        await page.evaluate(f"window.scrollBy(0, {random.randint(-150, -30)})")
        await asyncio.sleep(random.uniform(0.1, 0.3))


async def extract_article_content(page, article: dict) -> KnowledgeItem | None:
    """用 Playwright 提取单篇文章正文"""
    url = article["url"]
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await human_delay(0.5, 1.5)
        await simulate_scroll(page)

        data = await page.evaluate("""
            () => {
                const h1 = document.querySelector('h1');
                const title = h1 ? h1.textContent.trim() : '';

                // DXY 文章正文容器
                const articleEl = document.querySelector('[class*="article"]') ||
                                  document.querySelector('article') ||
                                  document.querySelector('.rich-text') ||
                                  document.querySelector('.content');
                const content = articleEl ? articleEl.textContent.trim() : '';

                return { title, content };
            }
        """)

        title = data.get("title", "").strip() or article.get("title", "")
        content = data.get("content", "").strip()

        if not title or len(content) < 200:
            return None
        if chinese_char_ratio(content) < 0.4:
            return None

        return KnowledgeItem(
            source="dxy",
            type="article",
            title=title,
            content=content,
            age_range=guess_age_range(title + content[:500]),
            category=guess_category(title + content[:500]),
            url=url,
        )

    except Exception as e:
        print(f"    [EXTRACT ERROR] {url}: {e}")
        return None


async def crawl_articles_with_playwright(articles: list[dict], auth_data: dict | None,
                                          max_articles: int = None) -> int:
    """Playwright 批量提取文章正文"""
    if max_articles:
        articles = articles[:max_articles]

    print(f"\n[CRAWL] Extracting content from {len(articles)} articles via Playwright...")

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": random.randint(1280, 1440), "height": random.randint(800, 960)},
            locale="zh-CN", timezone_id="Asia/Shanghai",
        )
        await context.add_init_script(ANTI_DETECTION_SCRIPT)

        # 注入 auth cookies
        if auth_data and auth_data.get("cookies"):
            await context.add_cookies(auth_data["cookies"])
            print(f"[AUTH] Injected {len(auth_data['cookies'])} cookies")

        page = await context.new_page()

        saved = 0
        for i, article in enumerate(articles):
            print(f"  [{i+1}/{len(articles)}] [{article['id']}] {article['title'][:60]}")

            item = await extract_article_content(page, article)

            if item and is_valid_knowledge(item):
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                saved += 1
                print(f"    [OK] {len(item.content)} chars | {item.age_range} | {item.category}")
            else:
                print(f"    [SKIP] No valid content")

            # 真人节奏
            await human_delay(0.8, 2.5)
            if (i + 1) % random.choice([8, 12, 15]) == 0:
                rest = random.uniform(4.0, 8.0)
                print(f"    --- Rest {rest:.0f}s ---")
                await asyncio.sleep(rest)

        await context.close()
        await browser.close()

    return saved


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

async def main_async(start_id: int = SCAN_START, end_id: int = SCAN_END,
                     max_articles: int = None, scan_only: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # 加载认证
    auth_file = os.path.join(AUTH_DIR, "dxy_auth.json")
    auth_data = None
    if os.path.exists(auth_file):
        with open(auth_file, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
        print(f"[AUTH] Loaded DXY auth: {len(auth_data.get('cookies', []))} cookies, "
              f"{len(auth_data.get('local_storage', {}))} localStorage keys")
    else:
        print("[AUTH] No auth file found, crawling without login")

    # Phase 1: 扫描文章 ID
    if os.path.exists(SCAN_CACHE_FILE):
        with open(SCAN_CACHE_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"[SCAN] Loaded {len(articles)} cached article IDs from {SCAN_CACHE_FILE}")
    else:
        articles = await scan_article_ids(start_id, end_id)
        with open(SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"[SCAN] Saved {len(articles)} article IDs to {SCAN_CACHE_FILE}")

    if max_articles:
        articles = articles[:max_articles]

    scan_elapsed = time.time() - t0
    print(f"\n[PHASE 1 DONE] {len(articles)} parenting articles found in {scan_elapsed:.0f}s")

    if scan_only:
        print("[SCAN ONLY] Skipping content extraction.")
        return

    if not articles:
        print("[DONE] No parenting articles to crawl.")
        return

    # Phase 2: Playwright 提取正文
    saved = await crawl_articles_with_playwright(articles, auth_data, max_articles)

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DXY 爬取完成: {saved}/{len(articles)} 篇")
    print(f"  总耗时: {total_elapsed:.0f}s")
    print(f"  输出: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DXY 丁香妈妈文章爬虫")
    parser.add_argument("--start", type=int, default=SCAN_START,
                        help=f"扫描起始文章 ID (默认: {SCAN_START})")
    parser.add_argument("--end", type=int, default=SCAN_END,
                        help=f"扫描结束文章 ID (默认: {SCAN_END})")
    parser.add_argument("--max", type=int, default=None,
                        help="最大爬取文章数")
    parser.add_argument("--scan-only", action="store_true",
                        help="仅扫描不爬取")
    args = parser.parse_args()

    asyncio.run(main_async(args.start, args.end, args.max, args.scan_only))


if __name__ == "__main__":
    main()
