"""爬取丁香妈妈公开文章 (dxy.cn)"""

import json
import os
import re
import sys
import time
import hashlib
from datetime import datetime
from urllib.parse import urljoin

import httpx
from lxml import html as lxml_html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_articles.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

SEED_URLS = [
    # 丁香妈妈 育儿知识列表页
    "https://dxy.com/column/304",
    "https://dxy.com/column/303",
    # 丁香妈妈 百科
    "https://dxy.com/faq/304",
]


def fetch_html(client: httpx.Client, url: str, retries: int = 3) -> str:
    """获取页面 HTML"""
    for attempt in range(retries):
        try:
            resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"  [RETRY {attempt + 1}/{retries}] {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return ""


def parse_article_list(html_text: str, base_url: str) -> list[dict]:
    """从列表页解析文章链接"""
    articles = []
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return articles

    # 常见的文章链接模式
    links = tree.xpath('//a[@href]')
    for link in links:
        href = link.get("href", "")
        title = (link.text_content() or "").strip()

        # 过滤出文章详情页链接
        if not title or len(title) < 2:
            continue

        full_url = urljoin(base_url, href)

        # 丁香妈妈文章 URL 模式
        if not any(pat in full_url for pat in [
            "/article/", "/column/", "/faq/", "/detail/", "/view/"
        ]):
            continue

        articles.append({"title": title, "url": full_url})

    return articles


def parse_article(html_text: str, url: str) -> dict | None:
    """解析文章详情页"""
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return None

    # 尝试多种正文提取策略
    title = ""
    content = ""

    # 标题
    title_candidates = tree.xpath('//h1 | //*[contains(@class, "title")]/text() | //*[contains(@class, "article-title")]')
    if title_candidates:
        title = (title_candidates[0].text_content() if hasattr(title_candidates[0], 'text_content')
                 else str(title_candidates[0])).strip()

    # 正文：优先用 article 标签
    article_body = tree.xpath('//article')
    if article_body:
        content = article_body[0].text_content()
    else:
        # 尝试常见的内容容器
        content_div = tree.xpath(
            '//*[contains(@class, "article-content")] | '
            '//*[contains(@class, "content")] | '
            '//*[contains(@class, "detail-content")] | '
            '//*[contains(@class, "post-content")] | '
            '//div[@id="content"]'
        )
        if content_div:
            content = content_div[0].text_content()
        else:
            # 最后手段：取 body 文本
            body = tree.xpath('//body')
            if body:
                content = body[0].text_content()

    # 清洗
    title = re.sub(r'\s+', ' ', title).strip()
    content = re.sub(r'\s+', ' ', content).strip()

    # 过滤太短或非中文内容
    if not title or len(content) < 100:
        return None
    if chinese_char_ratio(content) < 0.5:
        return None

    return {"title": title, "content": content, "url": url}


def crawl_seed_urls(client: httpx.Client) -> list[dict]:
    """从种子 URL 发现文章"""
    all_articles = []
    seen_urls = set()

    for seed_url in SEED_URLS:
        print(f"[CRAWL] Seed list: {seed_url}")
        html_text = fetch_html(client, seed_url)
        if not html_text:
            print(f"  [SKIP] Could not fetch {seed_url}")
            continue

        articles = parse_article_list(html_text, seed_url)
        print(f"  [FOUND] {len(articles)} article links")

        for art in articles:
            url = art["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            all_articles.append(art)

    return all_articles


def crawl_articles(client: httpx.Client, article_links: list[dict]) -> list[KnowledgeItem]:
    """逐篇爬取文章内容"""
    items = []
    for i, art in enumerate(article_links):
        url = art["url"]
        print(f"[{i + 1}/{len(article_links)}] {art['title'][:50]}...")

        html_text = fetch_html(client, url)
        if not html_text:
            continue

        parsed = parse_article(html_text, url)
        if not parsed:
            print(f"  [SKIP] Could not parse content")
            continue

        item = KnowledgeItem(
            source="dxy",
            type="article",
            title=parsed["title"],
            content=parsed["content"],
            age_range=guess_age_range(parsed["title"] + parsed["content"]),
            category=guess_category(parsed["title"] + parsed["content"]),
            url=url,
        )

        if is_valid_knowledge(item):
            items.append(item)
            print(f"  [OK] {len(item.content)} chars, age={item.age_range}, cat={item.category}")

        time.sleep(1)  # 礼貌爬取

    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with httpx.Client(timeout=30) as client:
        # Step 1: 发现文章链接
        article_links = crawl_seed_urls(client)
        print(f"\n[SUMMARY] Total articles to crawl: {len(article_links)}")

        # Step 2: 逐篇爬取
        items = crawl_articles(client, article_links)

        # Step 3: 保存
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

        print(f"\n[DONE] Saved {len(items)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
