"""
MATINF 数据集手动下载指南

MATINF 需要手动申请下载（Google 表单），不可程序化下载。

下载步骤:
  1. 访问 Google 表单: https://forms.gle/nkH4LVE4iNQeDzsc9
  2. 填写申请（姓名、机构、用途）
  3. 收到邮件后下载 zip 文件（含 train.csv / test.csv / dev.csv）
  4. 解压到 data/raw/matinf/ 目录
  5. 运行本脚本进行转换

本脚本读取解压后的 MATINF CSV 文件，转为统一 Schema 的 JSONL。
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category, is_valid_knowledge
)

MATINF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matinf")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matinf.jsonl")

AGE_MAP = {
    "0-1岁": "0-1月",
    "1-2岁": "1-3岁",
    "2-3岁": "1-3岁",
}

TOPIC_MAP = {
    "产褥期保健": "产后护理",
    "儿童过敏": "疾病",
    "动作发育": "发育",
    "婴幼保健": "健康",
    "婴幼心理": "心理",
    "婴幼早教": "早教",
    "婴幼期喂养": "喂养",
    "婴幼营养": "营养",
    "孕期保健": "孕期保健",
    "家庭教育": "家庭教育",
    "幼儿园": "幼儿园",
    "未准父母": "备孕",
    "流产和不孕": "孕期保健",
    "疫苗接种": "疫苗",
    "皮肤护理": "皮肤护理",
    "宝宝上火": "常见病",
    "腹泻": "疾病",
    "婴幼常见病": "疾病",
}


def load_matinf_csv(filepath: str) -> list[dict]:
    """加载 MATINF CSV 文件"""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def convert_to_knowledge_item(row: dict) -> KnowledgeItem | None:
    """将 MATINF 行转为 KnowledgeItem"""
    question = row.get("question", "").strip()
    description = row.get("description", "").strip()
    answer = row.get("answer", "").strip()
    label = row.get("class", "").strip()

    # 内容 = question + description（MATINF 核心是 QA 对，description 是问题描述）
    content = answer if answer else description
    if not content:
        return None

    # 判断年龄段和主题
    age_range = AGE_MAP.get(label, guess_age_range(question + description))
    category = TOPIC_MAP.get(label, guess_category(question + description))

    # 标题取 question
    title = question if question else (description[:100] if description else "")

    item = KnowledgeItem(
        source="matinf",
        type="qa",
        title=title,
        question=question,
        content=content,
        age_range=age_range,
        category=category,
        url="",
    )

    if is_valid_knowledge(item):
        return item
    return None


def main():
    if not os.path.exists(MATINF_DIR):
        print("=" * 60)
        print("  MATINF 数据集需要手动下载")
        print("=" * 60)
        print()
        print("  下载步骤:")
        print("  1. 访问: https://forms.gle/nkH4LVE4iNQeDzsc9")
        print("  2. 填写申请表单")
        print("  3. 收到邮件后下载 zip 文件")
        print("  4. 解压 train.csv / test.csv / dev.csv 到:")
        print(f"     {MATINF_DIR}")
        print("  5. 重新运行本脚本")
        print()
        print("=" * 60)
        # 创建目录占位
        os.makedirs(MATINF_DIR, exist_ok=True)
        return

    all_rows = []
    for csv_file in ["train.csv", "test.csv", "dev.csv"]:
        filepath = os.path.join(MATINF_DIR, csv_file)
        if os.path.exists(filepath):
            rows = load_matinf_csv(filepath)
            print(f"[LOAD] {csv_file}: {len(rows)} rows")
            all_rows.extend(rows)
        else:
            print(f"[WARN] {csv_file} not found in {MATINF_DIR}")

    if not all_rows:
        print("[ERROR] No CSV files found. Please download MATINF first.")
        return

    print(f"[INFO] Total rows: {len(all_rows)}")
    converted = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in all_rows:
            item = convert_to_knowledge_item(row)
            if item:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                converted += 1
            else:
                skipped += 1

    print(f"[DONE] Converted: {converted}, Skipped: {skipped}")
    print(f"[DONE] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
