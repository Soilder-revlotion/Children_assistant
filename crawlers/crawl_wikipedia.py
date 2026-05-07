"""
Wikipedia 育儿知识爬虫

通过 Wikipedia API 获取育儿相关页面，稳定可靠，无反爬。
"""

import json
import os
import sys
import time
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wikipedia_articles.jsonl")

# 育儿相关的 Wikipedia 页面标题（中文，通过搜索验证有实际内容）
ZH_PARENTING_TITLES = [
    # 孕期
    "妊娠", "分娩", "孕期", "产前检查",
    "产后护理", "坐月子", "产后抑郁症",
    # 婴儿
    "婴儿", "新生儿", "早产儿",
    "婴儿护理", "婴儿猝死症", "婴儿哭闹",
    "婴儿食品", "辅食",
    # 喂养
    "母乳哺育", "母乳喂养", "配方奶粉", "哺乳", "断奶",
    # 发育
    "儿童发育", "儿童发展阶段", "认知发展论",
    "语言发展", "精细动作", "幼儿",
    # 健康
    "儿童健康", "小儿发热", "小儿肺炎",
    "婴儿湿疹", "婴儿腹泻", "婴儿黄疸",
    "儿童肥胖", "维生素D缺乏症",
    "疫苗接种", "兒童疫苗接種計畫",
    # 教育
    "学前教育", "家庭教育", "蒙台梭利教育法",
    "幼儿园", "早期教育",
    # 心理
    "儿童心理学", "依恋理论", "分离焦虑障碍",
    # 安全
    "儿童安全", "儿童安全座椅",
    # 营养
    "婴幼儿营养", "营养不良",
    # 疾病
    "手足口病", "水痘", "麻疹", "百日咳",
    "小儿感冒", "肺炎", "腹泻病",
    # 综合
    "育儿", "母婴", "亲子关系",
]

EN_PARENTING_TITLES = [
    # Core parenting
    "Parenting", "Infant", "Toddler", "Child development",
    "Breastfeeding", "Infant formula", "Baby food",
    "Child development stages",
    # Health
    "Infant mortality", "Sudden infant death syndrome",
    "Vaccination", "Childhood immunization",
    "Common cold", "Fever",
    # Pregnancy
    "Pregnancy", "Prenatal care", "Childbirth",
    "Postpartum period",
    # Education
    "Early childhood education", "Montessori education",
    # Safety
    "Child safety seat",
]


def fetch_page_content(client: httpx.Client, title: str, lang: str = "zh") -> dict | None:
    """通过 Wikipedia REST API 获取页面内容"""
    base_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + title.replace(" ", "_")

    try:
        resp = client.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "title": data.get("title", title),
            "content": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "lang": lang,
            "description": data.get("description", ""),
        }
    except Exception:
        return None


def fetch_full_page(client: httpx.Client, title: str, lang: str = "zh") -> dict | None:
    """通过 Wikipedia API 获取页面全文（纯文本），自动处理重定向"""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "exintro": "0",         # 全文而非仅简介
        "explaintext": "1",
        "inprop": "url",
        "redirects": "1",       # 自动解决重定向
    }
    url = f"https://{lang}.wikipedia.org/w/api.php"

    try:
        resp = client.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                continue
            return {
                "title": page_data.get("title", title),
                "content": page_data.get("extract", ""),
                "url": page_data.get("fullurl", ""),
                "lang": lang,
                "page_id": page_id,
            }
    except Exception:
        return None


def crawl_wikipedia_titles(titles: list[str], lang: str = "zh") -> list[KnowledgeItem]:
    """批量获取 Wikipedia 页面"""
    items = []
    with httpx.Client(timeout=20) as client:
        for i, title in enumerate(titles):
            print(f"  [{i + 1}/{len(titles)}] [{lang}] {title}")

            # 使用 summary API (REST)
            data = fetch_page_content(client, title, lang)
            if not data or not data["content"]:
                # 降级到 full page API
                data = fetch_full_page(client, title, lang)

            if not data or not data["content"] or len(data["content"]) < 100:
                print(f"    [SKIP] No content or too short")
                continue

            content = data["content"]
            item_title = data["title"]

            if chinese_char_ratio(content) < 0.3 and lang == "zh":
                print(f"    [SKIP] Low Chinese ratio: {chinese_char_ratio(content):.2f}")
                continue

            item = KnowledgeItem(
                source=f"wikipedia_{lang}",
                type="knowledge",
                title=item_title,
                content=content,
                age_range=guess_age_range(item_title + content),
                category=guess_category(item_title + content),
                url=data.get("url", ""),
            )

            if is_valid_knowledge(item):
                items.append(item)
                print(f"    [OK] {len(content)} chars | {item.age_range} | {item.category}")
            else:
                print(f"    [SKIP] Failed validation")

            time.sleep(0.5)  # Wikipedia API 有频率限制

    return items


def search_wikipedia(client: httpx.Client, query: str, lang: str = "zh") -> list[str]:
    """搜索 Wikipedia 获取相关页面标题"""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 10,
    }
    url = f"https://{lang}.wikipedia.org/w/api.php"

    try:
        resp = client.get(url, params=params, timeout=10)
        data = resp.json()
        return [r["title"] for r in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def expand_titles_via_search(existing_titles: list[str]) -> list[str]:
    """通过 Wikipedia 搜索扩展相关页面"""
    search_queries = [
        "婴儿护理", "幼儿教育", "儿童疾病", "孕妇饮食",
        "产后恢复", "婴幼儿睡眠", "儿童心理", "亲子关系",
        "儿童营养", "婴儿发育", "早期教育方法",
        "child development", "baby care", "parenting tips",
    ]

    all_titles = set(existing_titles)
    with httpx.Client(timeout=15) as client:
        for query in search_queries:
            lang = "zh" if any('一' <= c <= '鿿' for c in query) else "en"
            print(f"  [SEARCH] [{lang}] {query}")
            titles = search_wikipedia(client, query, lang)
            for t in titles:
                all_titles.add(t)
            print(f"    Found {len(titles)} results")
            time.sleep(0.3)

    return list(all_titles)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 合并中英文标题
    all_titles = list(set(ZH_PARENTING_TITLES))
    print(f"[INFO] Base titles: {len(all_titles)}")

    # 可选：通过搜索扩展
    print("[INFO] Expanding titles via search...")
    all_titles = expand_titles_via_search(all_titles)
    print(f"[INFO] Expanded titles: {len(all_titles)}")

    # 爬取中文页面
    zh_titles = [t for t in all_titles if any('一' <= c <= '鿿' for c in t)]
    # 分离中英文
    en_titles = [t for t in all_titles if t not in zh_titles]

    all_items = []

    if zh_titles:
        print(f"\n[PHASE 1] Crawling {len(zh_titles)} Chinese Wikipedia pages...")
        items = crawl_wikipedia_titles(zh_titles, "zh")
        all_items.extend(items)
        print(f"  Collected {len(items)} Chinese articles")

    if en_titles:
        print(f"\n[PHASE 2] Crawling {len(en_titles)} English Wikipedia pages...")
        items = crawl_wikipedia_titles(en_titles, "en")
        all_items.extend(items)
        print(f"  Collected {len(items)} English articles")

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Total articles: {len(all_items)}")
    print(f"[DONE] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
