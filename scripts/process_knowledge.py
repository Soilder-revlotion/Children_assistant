"""数据处理管道：合并、清洗、去重、PII 脱敏、输出知识库"""

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, is_valid_knowledge, chinese_char_ratio
)

# PII 脱敏正则
_PII_PATTERNS = [
    # 中国手机号 (1xx-xxxx-xxxx)
    (re.compile(r'(?<!\d)(1[3-9]\d)(-?\d{4})(-?\d{4})(?!\d)'),
     r'\1****\3'),
    # 身份证号 (18位 / 15位)
    (re.compile(r'(?<!\d)(\d{6})(\d{4})(\d{4})(\d{3}[\dXx])(?!\d)'),
     r'\1****\3\4'),
    (re.compile(r'(?<!\d)(\d{6})(\d{6})(\d{3})(?!\d)'),
     r'\1****\3'),
    # 邮箱
    (re.compile(r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
     r'[EMAIL]'),
    # 固话 (0xx-xxxxxxxx / 0xxx-xxxxxxxx)
    (re.compile(r'(?<!\d)(0\d{2,3}-?\d{3,4})(\d{4})(?!\d)'),
     r'\1****'),
    # 微信号 (wxid_xxx / 微信: xxx / 微信号: xxx / VX: xxx)
    (re.compile(r'(?:微信|VX|vx|WeChat|wechat)[：:\s]*[a-zA-Z0-9_-]{6,20}'),
     r'[WECHAT]'),
    (re.compile(r'wxid_[a-zA-Z0-9_-]{6,20}'),
     r'[WECHAT]'),
    # QQ号 (5-12位数字，常以 QQ: 开头)
    (re.compile(r'(?:QQ|qq)[：:\s]*\d{5,12}'),
     r'[QQ]'),
    # 中国家庭地址（省市区开头 + 详细地址模式）
    (re.compile(
        r'(?:北京市|天津市|上海市|重庆市|'
        r'(?:河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|'
        r'河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|'
        r'内蒙古|广西|西藏|宁夏|新疆)'
        r'(?:省|自治区))'
        r'(?:[^\s,，。.]{2,50}(?:区|县|市|镇|乡|村|街道|路|街|巷|号|楼|栋|单元|室|层)){1,4}'),
     r'[ADDRESS]'),
    # 车牌号
    (re.compile(r'(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳])'),
     r'[PLATE]'),
    # IP 地址
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
     r'[IP]'),
    # URL（非必要 — 保留 dxy.com / baike.baidu.com 等来源链接，只脱敏含 token 的）
    (re.compile(r'(?:https?://)?(?:www\.)?(?!(?:dxy|babytree|baike|zhihu|mama)\.)[a-zA-Z0-9.-]+\.(?:com|cn|net|org)/\S*(?:token|key|secret|password|auth)\S*'),
     r'[SENSITIVE_URL]'),
]


def mask_pii(text: str) -> tuple[str, int]:
    """对文本执行 PII 脱敏，返回 (脱敏后文本, 脱敏处数)"""
    masked = text
    count = 0
    for pattern, replacement in _PII_PATTERNS:
        new_text, n = pattern.subn(replacement, masked)
        if n > 0:
            count += n
            masked = new_text
    return masked, count


def pii_masking(records: list[dict]) -> list[dict]:
    """批量 PII 脱敏"""
    total_masked = 0
    for rec in records:
        # 脱敏 content
        if rec.get("content"):
            masked, n = mask_pii(rec["content"])
            rec["content"] = masked
            total_masked += n
        # 脱敏 question
        if rec.get("question"):
            masked, n = mask_pii(rec["question"])
            rec["question"] = masked
            total_masked += n
        # 脱敏 title（title 可能也含 PII）
        if rec.get("title"):
            masked, n = mask_pii(rec["title"])
            rec["title"] = masked
            total_masked += n
    print(f"  [PII] Masked {total_masked} occurrences across {len(records)} records")
    return records

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


def load_all_raw() -> list[dict]:
    """加载所有原始数据文件"""
    all_records = []
    if not os.path.exists(RAW_DIR):
        return all_records

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(RAW_DIR, filename)
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_records.append(record)
                    count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  [LOAD] {filename}: {count} records")

    return all_records


def deduplicate(records: list[dict]) -> list[dict]:
    """去重：先 URL 去重，再内容 hash 去重"""
    seen_urls = set()
    seen_hashes = set()
    deduped = []

    for rec in records:
        # URL 去重
        url = rec.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        # 内容 hash 去重
        content = rec.get("content", "")
        if not content:
            continue
        content_hash = rec.get("content_hash", "")
        if not content_hash:
            import hashlib
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        deduped.append(rec)

    return deduped


def filter_quality(records: list[dict]) -> list[dict]:
    """质量过滤"""
    filtered = []
    for rec in records:
        content = rec.get("content", "")
        source = rec.get("source", "")
        # 长度检查
        if len(content) < 100:
            continue
        # 中文比例检查（仅对中文来源）
        if source.startswith("wikipedia_zh") or source in ("dxy", "babytree", "mama_cn", "ci123"):
            if chinese_char_ratio(content) < 0.3:
                continue
        # 构建临时 KnowledgeItem 做完整检查
        item = KnowledgeItem(
            id=rec.get("id", ""),
            source=rec.get("source", ""),
            type=rec.get("type", ""),
            title=rec.get("title", ""),
            question=rec.get("question", ""),
            content=content,
            age_range=rec.get("age_range", "通用"),
            category=rec.get("category", "其他"),
            url=rec.get("url", ""),
        )
        if is_valid_knowledge(item):
            filtered.append(rec)

    return filtered


def build_knowledge_base(records: list[dict]):
    """构建知识库：按年龄段 × 分类组织，输出统计"""
    os.makedirs(KB_DIR, exist_ok=True)

    # 全量 JSONL
    all_file = os.path.join(KB_DIR, "parenting_knowledge_base.jsonl")
    with open(all_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 按年龄段分组
    by_age = defaultdict(list)
    for rec in records:
        age = rec.get("age_range", "通用")
        by_age[age].append(rec)

    for age, items in sorted(by_age.items()):
        age_dir = os.path.join(KB_DIR, "by_age")
        os.makedirs(age_dir, exist_ok=True)
        safe_name = age.replace("/", "-").replace("+", "plus")
        filepath = os.path.join(age_dir, f"{safe_name}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 按分类分组
    by_cat = defaultdict(list)
    for rec in records:
        cat = rec.get("category", "其他")
        by_cat[cat].append(rec)

    for cat, items in sorted(by_cat.items()):
        cat_dir = os.path.join(KB_DIR, "by_category")
        os.makedirs(cat_dir, exist_ok=True)
        filepath = os.path.join(cat_dir, f"{cat}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return by_age, by_cat


def print_stats(records: list[dict], by_age: dict, by_cat: dict):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print(f"  知识库统计")
    print("=" * 60)
    print(f"  总条目数: {len(records)}")

    # 来源分布
    by_source = defaultdict(int)
    for rec in records:
        by_source[rec.get("source", "unknown")] += 1
    print(f"\n  来源分布:")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt}")

    # 年龄段分布
    print(f"\n  年龄段分布:")
    for age, items in sorted(by_age.items()):
        print(f"    {age}: {len(items)}")

    # 分类分布
    print(f"\n  分类分布 (Top 10):")
    top_cats = sorted(by_cat.items(), key=lambda x: -len(x[1]))[:10]
    for cat, items in top_cats:
        print(f"    {cat}: {len(items)}")

    print("=" * 60)


def main():
    print("[STEP 1] Loading raw data...")
    records = load_all_raw()
    print(f"  Total loaded: {len(records)}")

    if len(records) == 0:
        print("[WARN] No raw data found. Run crawlers first.")
        return

    print("\n[STEP 2] Deduplication...")
    records = deduplicate(records)
    print(f"  After dedup: {len(records)}")

    print("\n[STEP 3] PII masking...")
    records = pii_masking(records)

    print("\n[STEP 4] Quality filter...")
    records = filter_quality(records)
    print(f"  After filter: {len(records)}")

    print("\n[STEP 5] Building knowledge base...")
    by_age, by_cat = build_knowledge_base(records)

    print_stats(records, by_age, by_cat)

    # 保存统计报告
    report = {
        "total": len(records),
        "by_source": {},
        "by_age": {k: len(v) for k, v in by_age.items()},
        "by_category": {k: len(v) for k, v in by_cat.items()},
    }
    for rec in records:
        src = rec.get("source", "unknown")
        report["by_source"][src] = report["by_source"].get(src, 0) + 1

    report_path = os.path.join(PROCESSED_DIR, "stats.json")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
