"""
通用育儿网站爬虫

自动发现 + trafilatura 正文提取，支持:
  - 丁香妈妈 (dxy.com)
  - 宝宝树 (babytree.com)
  - 妈妈网 (mama.cn)
  - 知乎育儿话题
  - 百度百科育儿子类
"""

import json
import os
import re
import sys
import time
import hashlib
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from lxml import html as lxml_html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crawled_articles.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 各网站的种子 URL 和文章链接提取规则
SITE_CONFIGS = [
    {
        "name": "dxy",
        "seed_urls": [
            "https://dxy.com/column/304",   # 丁香妈妈育儿
            "https://dxy.com/faq/304",
        ],
        "article_patterns": [r"/article/\d+", r"/faq/\d+", r"/column/\d+"],
        "domain": "dxy.com",
    },
    {
        "name": "babytree",
        "seed_urls": [
            "https://www.babytree.com/community/",
            "https://www.babytree.com/learn/",
        ],
        "article_patterns": [r"/community/.*/\d+", r"/learn/.*/\d+"],
        "domain": "babytree.com",
    },
    {
        "name": "mama_cn",
        "seed_urls": [
            "https://www.mama.cn/",
            "https://yuer.mama.cn/",
        ],
        "article_patterns": [r"/article/\d+", r"/detail/\d+"],
        "domain": "mama.cn",
    },
    {
        "name": "zhihu_parenting",
        "seed_urls": [
            "https://www.zhihu.com/topic/19550931",  # 育儿话题
            "https://www.zhihu.com/topic/19634995",  # 母婴
        ],
        "article_patterns": [r"/question/\d+", r"/answer/\d+"],
        "domain": "zhihu.com",
    },
    {
        "name": "baike",
        "seed_urls": [
            "https://baike.baidu.com/item/育儿",
            "https://baike.baidu.com/item/婴儿护理",
            "https://baike.baidu.com/item/婴幼儿喂养",
            "https://baike.baidu.com/item/早期教育",
        ],
        "article_patterns": [r"/item/"],
        "domain": "baike.baidu.com",
    },
]

CRAWLED_URLS_FILE = os.path.join(OUTPUT_DIR, "crawled_urls.json")


def load_crawled_urls() -> set:
    """加载已爬取的 URL 集合"""
    if os.path.exists(CRAWLED_URLS_FILE):
        with open(CRAWLED_URLS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_crawled_urls(urls: set):
    """保存已爬取的 URL 集合"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CRAWLED_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(urls), f, ensure_ascii=False)


def is_article_url(url: str, patterns: list[str]) -> bool:
    """判断 URL 是否为文章页"""
    for pattern in patterns:
        if re.search(pattern, url):
            return True
    return False


def fetch_html(client: httpx.Client, url: str, retries: int = 2) -> str:
    """获取页面 HTML"""
    for attempt in range(retries):
        try:
            resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [FAIL] {url[:80]}: {e}")
            else:
                time.sleep(1.5)
    return ""


def extract_links(html_text: str, base_url: str, config: dict) -> list[str]:
    """从页面提取文章链接"""
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return []

    links = set()
    for a in tree.xpath('//a[@href]'):
        href = a.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("javascript"):
            continue

        full_url = urljoin(base_url, href)

        # 只保留同域名
        parsed = urlparse(full_url)
        if parsed.netloc != config["domain"] and not parsed.netloc.endswith("." + config["domain"]):
            continue

        # 匹配文章 URL 模式
        if is_article_url(full_url, config["article_patterns"]):
            links.add(full_url)

    return list(links)


def extract_content(html_text: str, url: str) -> dict | None:
    """使用 trafilatura 提取正文（降级到 lxml 自定义提取）"""
    # 首选：trafilatura
    try:
        import trafilatura
        doc = trafilatura.extract(
            html_text,
            url=url,
            include_comments=False,
            include_tables=False,
            output_format="dict",
            with_metadata=True,
        )
        if doc:
            title = doc.get("title", "")
            content = doc.get("text", "")
            if title and content and len(content) > 100 and chinese_char_ratio(content) > 0.4:
                return {"title": title, "content": content, "url": url}
    except Exception:
        pass

    # 降级：lxml 提取
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return None

    # 标题
    title = ""
    for xp in ['//h1', '//*[contains(@class, "title")]', '//title']:
        elems = tree.xpath(xp)
        if elems:
            text = (elems[0].text_content() if hasattr(elems[0], 'text_content')
                    else str(elems[0]))
            title = re.sub(r'\s+', ' ', text).strip()
            if len(title) > 2:
                break

    # 移除无用标签
    for tag in tree.xpath('//script | //style | //nav | //footer | //header | //aside'):
        tag.getparent().remove(tag)

    # 正文
    content = ""
    for xp in [
        '//article', '//*[@id="content"]', '//*[@class="content"]',
        '//*[contains(@class, "article")]', '//*[contains(@class, "post-body")]',
        '//*[contains(@class, "detail")]', '//div[@role="main"]',
    ]:
        elems = tree.xpath(xp)
        if elems:
            content = elems[0].text_content()
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 100:
                break

    if not title or len(content) < 100:
        return None
    if chinese_char_ratio(content) < 0.4:
        return None

    return {"title": title, "content": content, "url": url}


def crawl_site(client: httpx.Client, config: dict, crawled_urls: set) -> list[KnowledgeItem]:
    """爬取单个网站"""
    name = config["name"]
    print(f"\n{'=' * 50}")
    print(f"[SITE] {name}")
    print(f"{'=' * 50}")

    # Step 1: 从种子页发现文章链接
    all_article_links = set()
    for seed_url in config["seed_urls"]:
        print(f"  [SEED] {seed_url}")
        html_text = fetch_html(client, seed_url)
        if not html_text:
            print(f"    [SKIP] Could not fetch")
            continue

        links = extract_links(html_text, seed_url, config)
        new_links = [l for l in links if l not in crawled_urls]
        print(f"    [LINKS] Found {len(links)}, new: {len(new_links)}")
        all_article_links.update(new_links)

    print(f"  [TOTAL] {len(all_article_links)} new articles to crawl")

    # Step 2: 逐篇爬取
    items = []
    article_list = list(all_article_links)
    for i, url in enumerate(article_list):
        if url in crawled_urls:
            continue

        print(f"  [{i + 1}/{len(article_list)}] {url[:80]}")

        html_text = fetch_html(client, url)
        if not html_text:
            crawled_urls.add(url)
            continue

        extracted = extract_content(html_text, url)
        if not extracted:
            print(f"    [SKIP] No content extracted")
            crawled_urls.add(url)
            continue

        item = KnowledgeItem(
            source=name,
            type="article",
            title=extracted["title"],
            content=extracted["content"],
            age_range=guess_age_range(extracted["title"] + extracted["content"]),
            category=guess_category(extracted["title"] + extracted["content"]),
            url=url,
        )

        if is_valid_knowledge(item):
            items.append(item)
            print(f"    [OK] {len(item.content)} chars, age={item.age_range}, cat={item.category}")
        else:
            print(f"    [SKIP] Failed validation")

        crawled_urls.add(url)
        time.sleep(1)

    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    crawled_urls = load_crawled_urls()
    print(f"[INFO] Previously crawled: {len(crawled_urls)} URLs")

    all_items = []
    total_saved = 0

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for config in SITE_CONFIGS:
            try:
                items = crawl_site(client, config, crawled_urls)
                all_items.extend(items)

                # 增量保存
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                        total_saved += 1

                save_crawled_urls(crawled_urls)

            except Exception as e:
                print(f"  [ERROR] Site {config['name']}: {e}")
                continue

            # 站点间休息
            time.sleep(2)

    print(f"\n{'=' * 50}")
    print(f"[DONE] Total articles saved this run: {total_saved}")
    print(f"[DONE] Total URLs crawled: {len(crawled_urls)}")
    print(f"[DONE] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
