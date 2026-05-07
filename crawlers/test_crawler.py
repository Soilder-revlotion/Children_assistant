"""快速测试爬虫：验证各育儿网站内容提取效果"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import trafilatura
from lxml import html as lxml_html
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
}

TEST_URLS = [
    ("dxy", "https://dxy.com/"),
    ("dxy", "https://dxy.com/column/304"),
    ("babytree", "https://www.babytree.com/"),
    ("mama_cn", "https://www.mama.cn/"),
    ("ci123", "https://www.ci123.com/"),
]


def extract_with_trafilatura(html_text: str, url: str) -> dict | None:
    """使用 trafilatura 提取"""
    try:
        json_str = trafilatura.extract(
            html_text, url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )
        if json_str:
            data = json.loads(json_str)
            title = data.get("title", "")
            content = data.get("text", "")
            if title and len(content) > 80 and chinese_char_ratio(content) > 0.3:
                return {"title": title, "content": content, "url": url}
    except Exception as e:
        print(f"    trafilatura error: {e}")
    return None


def extract_with_lxml(html_text: str, url: str) -> dict | None:
    """使用 lxml 提取"""
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return None

    # 移除干扰标签
    for tag_name in ['script', 'style', 'nav', 'footer', 'header', 'aside']:
        for tag in tree.xpath(f'//{tag_name}'):
            try:
                tag.getparent().remove(tag)
            except Exception:
                pass

    # 标题
    title = ""
    for xp in ['//h1', '//title', '//*[contains(@class, "title")]']:
        elems = tree.xpath(xp)
        for e in elems:
            t = e.text_content().strip()
            if len(t) > 2 and len(t) < 200:
                title = t
                break
        if title:
            break

    # 正文
    content = ""
    for xp in [
        '//article',
        '//div[@id="content"]',
        '//div[contains(@class, "content")]',
        '//div[contains(@class, "article")]',
        '//div[contains(@class, "post")]',
        '//main',
        '//body',
    ]:
        elems = tree.xpath(xp)
        if elems:
            content = elems[0].text_content()
            content = ' '.join(content.split())
            if len(content) > 100:
                break

    if not title or len(content) < 80:
        return None
    if chinese_char_ratio(content) < 0.3:
        return None

    return {"title": title, "content": content, "url": url}


def find_article_links(html_text: str, base_url: str, domain: str) -> list[str]:
    """从页面发现文章链接"""
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return []

    links = set()
    for a in tree.xpath('//a[@href]'):
        href = (a.get("href") or "").strip()
        text = (a.text_content() or "").strip()

        if not href or href.startswith("#") or href.startswith("javascript"):
            continue
        if not text or len(text) < 2:
            continue

        # 构建完整 URL
        if href.startswith("http"):
            full_url = href
        elif href.startswith("//"):
            full_url = "https:" + href
        elif href.startswith("/"):
            full_url = f"https://{domain}{href}"
        else:
            full_url = f"https://{domain}/{href}"

        # 只保留同域名
        if domain not in full_url:
            continue

        # 只看育儿相关链接
        text_lower = text.lower()
        is_parenting = any(kw in text_lower for kw in [
            "育儿", "宝宝", "婴儿", "幼儿", "孩子", "孕妇", "孕期",
            "喂养", "辅食", "睡眠", "发育", "早教", "疫苗", "母乳",
            "发烧", "咳嗽", "腹泻", "湿疹", "黄疸", "月子",
        ])
        if is_parenting and len(text) > 2:
            links.add(full_url)

    return list(links)


def test_url(client: httpx.Client, name: str, url: str):
    """测试单个 URL"""
    print(f"\n{'=' * 50}")
    print(f"[{name}] {url}")

    try:
        resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
    except Exception as e:
        print(f"  [FAIL] {e}")
        return None, []

    print(f"  HTTP {resp.status_code}, {len(resp.text)} chars")

    # 1. 尝试提取正文
    result = extract_with_trafilatura(resp.text, url)
    method = "trafilatura"
    if not result:
        result = extract_with_lxml(resp.text, url)
        method = "lxml"

    if result:
        print(f"  [{method}] title: {result['title'][:60]}")
        print(f"  [{method}] content: {len(result['content'])} chars, "
              f"chinese_ratio={chinese_char_ratio(result['content']):.2f}")
        print(f"  [{method}] preview: {result['content'][:150]}...")

        item = KnowledgeItem(
            source=name,
            type="article",
            title=result["title"],
            content=result["content"],
            age_range=guess_age_range(result["title"] + result["content"]),
            category=guess_category(result["title"] + result["content"]),
            url=url,
        )
        print(f"  [Schema] age={item.age_range}, cat={item.category}, "
              f"valid={is_valid_knowledge(item)}")
    else:
        print(f"  [SKIP] No content extracted")
        item = None

    # 2. 发现文章链接
    domain = url.split("://")[1].split("/")[0].replace("www.", "")
    article_links = find_article_links(resp.text, url, domain)
    parenting_links = len(article_links)
    print(f"  [Links] Found {parenting_links} parenting-related links")
    if parenting_links > 0 and parenting_links <= 5:
        for link in article_links[:5]:
            print(f"    - {link}")

    return item, article_links


def main():
    items = []
    all_links = []

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name, url in TEST_URLS:
            item, links = test_url(client, name, url)
            if item:
                items.append(item)
            all_links.extend(links)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Articles extracted: {len(items)}/{len(TEST_URLS)}")
    print(f"  Article links found: {len(all_links)}")
    for item in items:
        print(f"  [{item.source}] {item.title[:50]} | {item.age_range} | {item.category}")


if __name__ == "__main__":
    main()
