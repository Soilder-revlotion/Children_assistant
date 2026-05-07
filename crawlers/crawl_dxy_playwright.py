"""
丁香妈妈文章爬虫 (Playwright 真人模拟版)

使用 Playwright 加载栏目页提取文章链接（模拟真人浏览），
然后用 httpx + trafilatura 批量爬取文章内容。

用法:
  python crawlers/crawl_dxy_playwright.py
  python crawlers/crawl_dxy_playwright.py --max 50
"""

import json
import os
import random
import re
import sys
import time
import asyncio
import httpx
import trafilatura
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_articles.jsonl")
URLS_FILE = os.path.join(OUTPUT_DIR, "dxy_article_urls.json")

COLUMN_URLS = [
    ("https://dxy.com/column/304", "育儿知识"),
    ("https://dxy.com/column/303", "母婴健康"),
]

PARENTING_KW = [
    "宝宝", "婴儿", "幼儿", "儿童", "新生儿", "孩子", "小孩",
    "育儿", "母婴", "亲子", "孕期", "怀孕", "孕妇", "产妇",
    "月子", "产后", "备孕", "胎教", "母乳", "配方奶", "奶粉",
    "辅食", "断奶", "喂奶", "发烧", "感冒", "咳嗽", "腹泻",
    "湿疹", "过敏", "黄疸", "疫苗", "发育", "睡眠", "早教",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


async def human_delay(min_s=0.5, max_s=3.0):
    await asyncio.sleep(random.uniform(min_s, max_s))


async def simulate_scroll(page, scrolls=None):
    if scrolls is None:
        scrolls = random.randint(3, 6)
    for _ in range(scrolls):
        delta = random.randint(200, 600)
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(random.uniform(0.2, 0.8))
    if random.random() < 0.3:
        await page.evaluate(f"window.scrollBy(0, {random.randint(-200, -50)})")
        await asyncio.sleep(random.uniform(0.2, 0.4))


async def simulate_mouse(page):
    for _ in range(random.randint(1, 3)):
        await page.mouse.move(random.randint(100, 800), random.randint(100, 600),
                              steps=random.randint(3, 10))
        await asyncio.sleep(random.uniform(0.05, 0.15))


async def extract_article_links_from_column(page, column_url: str) -> set[str]:
    """用 Playwright 加载栏目页，提取文章链接（真人模拟）"""
    links = set()
    print(f"  [LOAD] {column_url}")

    try:
        await page.goto(column_url, wait_until="networkidle", timeout=30000)
        await human_delay(1.0, 2.5)
        await simulate_mouse(page)
        await simulate_scroll(page, random.randint(4, 7))

        # 提取所有文章链接
        hrefs = await page.evaluate("""
            () => {
                const links = document.querySelectorAll('a[href*="/article/"]');
                return Array.from(links).map(a => ({
                    href: a.getAttribute('href'),
                    text: a.textContent.trim().substring(0, 100)
                }));
            }
        """)

        for item in hrefs:
            href = item.get("href", "")
            text = item.get("text", "")
            if not href:
                continue

            # 构建完整 URL
            if href.startswith("/"):
                full_url = f"https://dxy.com{href}"
            elif href.startswith("http"):
                full_url = href
            else:
                continue

            # 提取文章 ID
            match = re.search(r"/article/(\d+)", full_url)
            if not match:
                continue

            # 过滤：只保留育儿相关的
            if text and any(kw in text for kw in PARENTING_KW):
                links.add(full_url)
            elif any(kw in href for kw in ["育儿", "baby", "母婴", "parent"]):
                links.add(full_url)

        print(f"    Found {len(links)} parenting article links")

    except Exception as e:
        print(f"    [ERROR] {e}")

    return links


async def discover_urls() -> list[str]:
    """发现所有育儿文章 URL（真人模拟）"""
    all_links = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ]
        )
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": random.randint(1280, 1440), "height": random.randint(800, 960)},
            locale="zh-CN", timezone_id="Asia/Shanghai",
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
        """)
        page = await context.new_page()

        for column_url, name in COLUMN_URLS:
            print(f"[DISCOVER] {name}: {column_url}")
            links = await extract_article_links_from_column(page, column_url)
            all_links.update(links)
            await human_delay(2.0, 5.0)  # 栏目间随机间隔

        await browser.close()

    print(f"\n[DISCOVER] Total unique URLs: {len(all_links)}")
    return list(all_links)


def crawl_articles(urls: list[str]) -> list[KnowledgeItem]:
    """用 httpx + trafilatura 批量爬取文章内容"""
    items = []
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://dxy.com/",
    }

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(urls):
            print(f"  [{i + 1}/{len(urls)}] {url[:80]}")
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code != 200:
                    print(f"    [SKIP] HTTP {resp.status_code}")
                    continue

                json_str = trafilatura.extract(
                    resp.text, url=url,
                    output_format="json",
                    with_metadata=True,
                    include_comments=False,
                    include_tables=False,
                )
                if not json_str:
                    print(f"    [SKIP] No content")
                    continue

                data = json.loads(json_str)
                title = data.get("title", "").strip()
                content = data.get("text", "").strip()

                if not title or len(content) < 200:
                    print(f"    [SKIP] Too short")
                    continue
                if chinese_char_ratio(content) < 0.5:
                    print(f"    [SKIP] Low Chinese ratio")
                    continue

                item = KnowledgeItem(
                    source="dxy",
                    type="article",
                    title=title,
                    content=content,
                    age_range=guess_age_range(title + content),
                    category=guess_category(title + content),
                    url=url,
                )

                if is_valid_knowledge(item):
                    items.append(item)
                    print(f"    [OK] {len(content)} chars | {item.age_range} | {item.category}")
                else:
                    print(f"    [SKIP] Failed validation")

            except Exception as e:
                print(f"    [ERROR] {e}")

            # 随机间隔（真人浏览节奏）
            time.sleep(random.uniform(0.5, 2.0))

            if (i + 1) % 20 == 0:
                rest = random.uniform(3.0, 6.0)
                print(f"  --- Rest {rest:.0f}s ---")
                time.sleep(rest)

    return items


def main():
    import argparse
    parser = argparse.ArgumentParser(description="丁香妈妈文章爬虫")
    parser.add_argument("--max", type=int, default=None, help="最大爬取文章数")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 用 Playwright 发现文章链接
    urls = asyncio.run(discover_urls())

    # 保存 URL 列表
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(urls, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {len(urls)} URLs to {URLS_FILE}")

    if not urls:
        print("[WARN] No URLs discovered. Trying fallback: known article IDs...")
        urls = [f"https://dxy.com/article/{aid}"
                for aid in range(199000, 201000)]

    if args.max:
        urls = urls[:args.max]

    # Step 2: 批量爬取文章
    print(f"\n[CRAWL] Processing {len(urls)} articles...")
    items = crawl_articles(urls)

    # Step 3: 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Saved {len(items)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
