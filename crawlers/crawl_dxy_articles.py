"""
丁香园文章批量爬虫

通过 ID 区间扫描获取育儿相关文章
文章 URL 格式: https://dxy.com/article/{id}
"""

import json
import os
import sys
import time
import httpx
import trafilatura

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_articles.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "dxy_progress.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 育儿相关关键词（用于过滤无关文章）
PARENTING_KEYWORDS = [
    # 核心育儿词
    "宝宝", "婴儿", "幼儿", "儿童", "新生儿", "孩子", "小孩",
    "育儿", "母婴", "亲子",
    # 年龄阶段
    "孕期", "怀孕", "孕妇", "产妇", "月子", "产后", "备孕", "分娩",
    "胎儿", "胎教", "胎动",
    # 喂养
    "母乳", "配方奶", "奶粉", "喂奶", "断奶", "辅食", "厌奶",
    "喂养", "奶瓶", "哺乳",
    # 健康/疾病
    "发烧", "感冒", "咳嗽", "腹泻", "湿疹", "过敏", "黄疸",
    "肺炎", "水痘", "手足口", "便秘", "呕吐", "腹痛",
    "疫苗", "接种", "预防针",
    # 发育
    "发育", "翻身", "爬行", "走路", "说话", "出牙", "长牙",
    "身高", "体重", "头围", "大运动", "精细动作",
    # 睡眠
    "睡眠", "睡觉", "哄睡", "夜醒", "入睡", "作息",
    # 早教/心理
    "早教", "启蒙", "绘本", "游戏", "安全感", "分离焦虑",
    # 护理
    "换尿布", "洗澡", "抚触", "脐带", "皮肤护理", "痱子",
    # 营养
    "维生素D", "钙", "铁", "锌", "DHA", "益生菌", "营养",
]


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_id": 1, "articles_found": 0, "ids_checked": 0}


def save_progress(progress: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)


def is_parenting_article(title: str, content: str) -> bool:
    """检查文章是否与育儿相关"""
    text = (title + " " + content[:500]).lower()
    for kw in PARENTING_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def fetch_article(client: httpx.Client, article_id: int) -> dict | None:
    """获取单篇文章"""
    url = f"https://dxy.com/article/{article_id}"
    try:
        resp = client.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    # trafilatura 提取
    try:
        json_str = trafilatura.extract(
            resp.text, url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )
        if not json_str:
            return None

        data = json.loads(json_str)
        title = data.get("title", "")
        content = data.get("text", "")

        if not title or len(content) < 200:
            return None
        if chinese_char_ratio(content) < 0.5:
            return None
        if not is_parenting_article(title, content):
            return None

        return {
            "url": url,
            "title": title.strip(),
            "content": content.strip(),
            "article_id": article_id,
        }
    except Exception:
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()

    # 扫描范围配置
    START_ID = progress["last_id"]
    END_ID = START_ID + 5000  # 每次运行扫描 5000 个 ID
    BATCH_SIZE = 20

    print(f"[INFO] Scanning article IDs {START_ID} to {END_ID}")
    print(f"[INFO] Previously found: {progress['articles_found']} articles")

    found_count = 0
    checked_count = 0

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for batch_start in range(START_ID, END_ID, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, END_ID)

            # 并发请求一个 batch
            results = []
            for aid in range(batch_start, batch_end):
                result = fetch_article(client, aid)
                checked_count += 1
                if result:
                    results.append(result)
                    found_count += 1

            # 保存批次结果
            if results:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for r in results:
                        item = KnowledgeItem(
                            source="dxy",
                            type="article",
                            title=r["title"],
                            content=r["content"],
                            age_range=guess_age_range(r["title"] + r["content"]),
                            category=guess_category(r["title"] + r["content"]),
                            url=r["url"],
                        )
                        if is_valid_knowledge(item):
                            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

            # 更新进度
            progress["last_id"] = batch_end
            progress["articles_found"] += len(results)
            progress["ids_checked"] = checked_count
            save_progress(progress)

            # 进度输出
            if (batch_start - START_ID) % 500 == 0 or results:
                print(f"  [{batch_end}/{END_ID}] checked={checked_count}, "
                      f"found={found_count}, latest_id={batch_end}")

            time.sleep(0.3)  # 礼貌间隔

    print(f"\n[DONE] Checked: {checked_count}, Found: {found_count}")
    print(f"[DONE] Progress saved. Next start ID: {progress['last_id']}")
    print(f"[DONE] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
