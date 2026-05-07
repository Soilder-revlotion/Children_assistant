"""
默沙东诊疗手册爬虫 (Playwright SSG/JS 渲染版)

MSD Manual 页面完全由 JavaScript 渲染，必须使用浏览器。

用法:
  python crawlers/crawl_msd_manual.py
  python crawlers/crawl_msd_manual.py --max 50
"""

import asyncio
import json
import os
import re
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "msd_manual.jsonl")
SITEMAP_FILE = os.path.join(OUTPUT_DIR, "msd_manual_urls.json")

START_URLS = [
    "https://www.msdmanuals.cn/home/children-s-health-issues",
    "https://www.msdmanuals.cn/home/women-s-health-issues",
    "https://www.msdmanuals.cn/home/infections/immunization",
]

PARENTING_KW = re.compile(
    "宝宝|婴儿|幼儿|儿童|新生儿|孩子|小孩|"
    "育儿|母婴|亲子|孕期|怀孕|孕妇|产妇|"
    "月子|产后|备孕|胎教|母乳|配方奶|奶粉|"
    "辅食|断奶|喂奶|发烧|感冒|咳嗽|腹泻|"
    "湿疹|过敏|黄疸|疫苗|发育|睡眠|早教|"
    "儿科|小儿|婴幼儿|学龄|入园|生长|"
    "妊娠|分娩|喂养|哺乳"
)

SKIP_PATTERNS = [r"/search", r"/about", r"/contact", r"/privacy", r"\.pdf$"]


def is_parenting(title: str, url: str) -> bool:
    return bool(PARENTING_KW.search(title + url))


async def discover_urls(page, start_url: str) -> list[tuple[str, str]]:
    """从 MSD Manual 板块页发现所有文章链接"""
    print(f"  [DISCOVER] {start_url}")
    await page.goto(start_url, wait_until="networkidle", timeout=30000)
    await asyncio.sleep(2)

    links = await page.evaluate("""
        () => {
            const links = document.querySelectorAll('a[href]');
            const results = [];
            const seen = new Set();
            links.forEach(a => {
                const href = a.getAttribute('href') || '';
                const text = (a.textContent || '').trim();
                if (!href || href.startsWith('#') || href.startsWith('javascript:')) return;
                if (text.length < 3) return;
                const key = href.split('#')[0];
                if (seen.has(key)) return;
                seen.add(key);
                results.push({href: key, text: text.slice(0, 100)});
            });
            return results;
        }
    """)

    articles = []
    for item in links:
        href = item["href"]
        text = item["text"]
        if not href or not text:
            continue
        if href.startswith("/"):
            href = f"https://www.msdmanuals.cn{href}"
        elif "msdmanuals.cn" not in href:
            continue
        href = re.sub(r"#.*$", "", href)
        if any(re.search(p, href) for p in SKIP_PATTERNS):
            continue
        if is_parenting(text, href):
            articles.append((href, text))

    return articles


async def extract_article(page, url: str) -> KnowledgeItem | None:
    """提取单篇文章"""
    try:
        await page.goto(url, wait_until="networkidle", timeout=25000)
        await asyncio.sleep(1)

        data = await page.evaluate("""
            () => {
                const h1 = document.querySelector('h1');
                const title = h1 ? h1.textContent.trim() : '';

                // 正文容器
                const main = document.querySelector('main') ||
                            document.querySelector('article') ||
                            document.querySelector('[class*="content"]') ||
                            document.querySelector('[class*="article"]') ||
                            document.querySelector('#content');
                const content = main ? main.textContent.trim() : document.body.textContent;

                // 去掉标题中的网站名
                const cleanTitle = title.replace(/\\s*[-–|]\\s*《?默沙东诊疗手册.*$/, '')
                                        .replace(/\\s*[-–|]\\s*MSD Manual.*$/, '');
                return {
                    title: cleanTitle,
                    content: content.replace(/\\s+/g, ' ').trim()
                };
            }
        """)

        title = data.get("title", "").strip()
        content = data.get("content", "").strip()

        if not title or len(content) < 200:
            return None
        if chinese_char_ratio(content) < 0.3:
            return None

        return KnowledgeItem(
            source="msd_manual",
            type="article",
            title=title[:200],
            content=content[:5000],
            age_range=guess_age_range(title + content[:500]),
            category=guess_category(title + content[:500]),
            url=url,
        )
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


async def main_async(max_articles: int = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="zh-CN", timezone_id="Asia/Shanghai",
        )
        page = await context.new_page()

        # Phase 1: URL discovery
        print("=" * 60)
        print("  默沙东诊疗手册 — 育儿内容爬虫 (Playwright)")
        print("=" * 60)
        print("\n[PHASE 1] URL discovery...")

        all_articles = []
        seen = set()
        for start_url in START_URLS:
            articles = await discover_urls(page, start_url)
            for url, title in articles:
                if url not in seen:
                    seen.add(url)
                    all_articles.append((url, title))
            print(f"    → {len(articles)} parenting articles from this section")

        print(f"\n  Total unique parenting articles: {len(all_articles)}")

        if max_articles:
            all_articles = all_articles[:max_articles]

        with open(SITEMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        # Phase 2: Extract content
        print(f"\n[PHASE 2] Extracting {len(all_articles)} articles...")
        saved = 0

        for i, (url, link_title) in enumerate(all_articles):
            print(f"  [{i+1}/{len(all_articles)}] {url[:100]}")

            item = await extract_article(page, url)
            if item and is_valid_knowledge(item):
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                saved += 1
                print(f"    [OK] {len(item.content)} chars | {item.age_range} | {item.category}")
            else:
                print(f"    [SKIP] No content")

            if (i + 1) % 20 == 0:
                print(f"    --- Progress: {saved}/{i+1} saved ---")

        await context.close()
        await browser.close()

    return saved, len(all_articles)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="默沙东诊疗手册爬虫")
    parser.add_argument("--max", type=int, default=None, help="最大文章数")
    args = parser.parse_args()

    t0 = time.time()
    saved, total = asyncio.run(main_async(args.max))
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  MSD Manual 完成: {saved}/{total} 篇 ({elapsed:.0f}s)")
    print(f"  输出: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
