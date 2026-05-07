"""
丁香园文章快速扫描器

异步并发扫描 ID 区间，通过 HTTP 状态码 + 标题关键词快速筛选育儿文章。
"""

import json
import os
import sys
import time
import asyncio
import httpx
import trafilatura

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_articles.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "dxy_scan_progress.json")

PARENTING_KW = [
    "宝宝", "婴儿", "幼儿", "儿童", "新生儿", "孩子", "小孩",
    "育儿", "母婴", "亲子", "孕期", "怀孕", "孕妇", "产妇",
    "月子", "产后", "备孕", "胎教", "母乳", "喂养", "奶粉",
    "辅食", "断奶", "喂奶", "哺乳", "发烧", "感冒", "咳嗽",
    "腹泻", "湿疹", "过敏", "黄疸", "疫苗", "发育", "睡眠",
    "早教", "启蒙", "安全感", "维生素D", "辅食添加",
]

CONCURRENCY = 10  # 并发数
BATCH_SIZE = 1000  # 每批扫描 1000 个 ID


def is_parenting_html(html_text: str) -> bool:
    """快速检查 HTML 是否可能是育儿文章"""
    # 取 title 和开头部分检查
    check_text = ""
    if "<title>" in html_text:
        title_start = html_text.find("<title>") + 7
        title_end = html_text.find("</title>", title_start)
        if title_end > title_start:
            check_text += html_text[title_start:title_end]
    check_text += " " + html_text[:2000]

    for kw in PARENTING_KW:
        if kw in check_text:
            return True
    return False


async def quick_check(client: httpx.AsyncClient, article_id: int) -> str | None:
    """快速检查文章是否存在且与育儿相关，返回 HTML"""
    url = f"https://dxy.com/article/{article_id}"
    try:
        resp = await client.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        html_text = resp.text
        if len(html_text) < 5000:
            return None
        if not is_parenting_html(html_text):
            return None
        return html_text
    except Exception:
        return None


def extract_article(html_text: str, article_id: int) -> KnowledgeItem | None:
    """从 HTML 提取文章"""
    url = f"https://dxy.com/article/{article_id}"
    try:
        json_str = trafilatura.extract(
            html_text, url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )
        if not json_str:
            return None

        data = json.loads(json_str)
        title = data.get("title", "").strip()
        content = data.get("text", "").strip()

        if not title or len(content) < 200:
            return None
        if chinese_char_ratio(content) < 0.5:
            return None

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
            return item
    except Exception:
        pass
    return None


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"next_id": 1, "found": 0, "checked": 0}


def save_progress(p: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f)


async def scan_range(start_id: int, end_id: int) -> list[KnowledgeItem]:
    """异步扫描 ID 区间"""
    items = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def scan_one(aid: int):
        async with semaphore:
            html_text = await quick_check(client, aid)
            if html_text:
                item = extract_article(html_text, aid)
                return item
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
        tasks = [scan_one(aid) for aid in range(start_id, end_id)]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            items.append(result)

    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()
    next_id = progress["next_id"]

    # 从已知有文章的区域开始
    if next_id == 1:
        next_id = 199000  # 已知的有效 ID 区域

    end_id = next_id + BATCH_SIZE

    print(f"[INFO] Scanning IDs {next_id} to {end_id}")
    print(f"[INFO] Previously found: {progress['found']}")

    items = asyncio.run(scan_range(next_id, end_id))

    # 保存结果
    if items:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    # 更新进度
    progress["next_id"] = end_id
    progress["found"] += len(items)
    progress["checked"] = end_id - next_id
    save_progress(progress)

    # 显示找到的文章
    print(f"\n[RESULTS] Found {len(items)} articles in {BATCH_SIZE} IDs")
    for item in items[:5]:
        print(f"  [{item.source}] {item.title[:60]} | {item.category} | {len(item.content)} chars")

    print(f"\n[DONE] Total found: {progress['found']}")
    print(f"[DONE] Next start ID: {end_id}")
    print(f"[DONE] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
