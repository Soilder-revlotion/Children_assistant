"""
知识库质量增强 — 精细化标签 + 实体识别 + 质量评分 + 知识缺口分析

用法:
  python scripts/enhance_knowledge.py              # 完整增强
  python scripts/enhance_knowledge.py --report-only  # 仅生成报告
  python scripts/enhance_knowledge.py --gap-analysis  # 仅知识缺口分析
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base",
                       "parenting_knowledge_base.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base",
                           "parenting_knowledge_base_enhanced.jsonl")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base",
                           "knowledge_quality_report.json")

# ─── 细粒度年龄段映射 ──────────────────────────────────
AGE_KEYWORDS_V2 = {
    "备孕": ["备孕", "孕前检查", "叶酸", "排卵", "精子", "备孕期", "准备怀孕"],
    "孕期": ["孕期", "怀孕", "孕妇", "产检", "胎动", "孕吐", "妊娠", "孕中期", "孕晚期",
             "唐筛", "糖耐", "胎心", "宫缩", "分娩", "顺产", "剖腹产"],
    "0-1月": ["新生儿", "满月", "月子", "初生", "刚出生", "脐带", "黄疸期", "初乳"],
    "1-6月": ["婴儿", "小月龄", "母乳", "配方奶", "胀气", "肠绞痛", "夜醒", "吐奶",
              "囟门", "追视", "抬头", "翻身训练", "百日"],
    "6-12月": ["辅食", "爬行", "出牙", "断奶", "学步", "咀嚼", "分离焦虑期",
               "六个月", "七个月", "八个月", "九个月", "十个月", "十一个月"],
    "1-3岁": ["幼儿", "学步", "说话", "如厕", "任性", "物权意识", "社交萌芽",
              "一岁", "两岁", "周岁", "挑食期", "秩序敏感"],
    "3-6岁": ["学龄前", "幼儿园", "入园", "社交", "识字", "专注力", "规则意识",
              "三岁", "四岁", "五岁", "自理能力"],
    "6岁+": ["小学", "上学", "学业", "青春期前", "六岁", "七岁", "八岁", "学习能力"],
}

# ─── 实体关键词库 ────────────────────────────────────────
ENTITY_PATTERNS = {
    "疾病": ["感冒", "发烧", "咳嗽", "腹泻", "湿疹", "过敏", "肺炎", "支气管炎", "哮喘",
             "手足口病", "水痘", "麻疹", "风疹", "腮腺炎", "猩红热", "流感", "鼻炎",
             "中耳炎", "扁桃体炎", "喉炎", "结膜炎", "鹅口疮", "尿布疹", "痱子",
             "黄疸", "贫血", "佝偻病", "肥胖", "营养不良", "自闭症", "多动症", "抑郁症"],
    "症状": ["流鼻涕", "鼻塞", "打喷嚏", "呕吐", "恶心", "腹痛", "便秘", "腹胀",
             "食欲不振", "哭闹", "烦躁", "嗜睡", "失眠", "出汗", "瘙痒", "皮疹",
             "红疹", "水疱", "脱皮", "口臭", "磨牙", "夜惊", "梦游"],
    "药物": ["布洛芬", "对乙酰氨基酚", "蒙脱石散", "益生菌", "维生素D", "维生素AD",
             "钙剂", "铁剂", "锌剂", "退烧药", "抗生素", "抗过敏药", "止咳药"],
    "人群": ["早产儿", "低体重儿", "巨大儿", "双胞胎", "过敏体质", "湿疹宝宝"],
    "机构": ["儿科", "儿保健科", "疾控中心", "妇幼保健院", "儿童医院"],
}

# ─── 知识缺口检查表 ─────────────────────────────────────
COVERAGE_CHECKS = [
    # (类别, 年龄段, 应有条目数阈值)
    ("喂养", "0-1月", 10),
    ("喂养", "1-6月", 15),
    ("喂养", "6-12月", 15),
    ("睡眠", "0-1月", 5),
    ("睡眠", "1-6月", 8),
    ("睡眠", "6-12月", 8),
    ("健康", "孕期", 10),
    ("疾病", "0-1月", 5),
    ("疾病", "1-6月", 8),
    ("疾病", "6-12月", 8),
    ("发育", "0-1月", 5),
    ("发育", "1-6月", 10),
    ("发育", "6-12月", 10),
    ("发育", "1-3岁", 10),
    ("早教", "1-3岁", 8),
    ("早教", "3-6岁", 8),
    ("疫苗", "通用", 15),
    ("孕期保健", "孕期", 15),
    ("产后护理", "0-1月", 10),
    ("安全", "1-6月", 5),
    ("安全", "6-12月", 5),
    ("安全", "1-3岁", 8),
    ("心理", "1-3岁", 5),
    ("心理", "3-6岁", 8),
    ("行为习惯", "1-3岁", 5),
    ("行为习惯", "3-6岁", 5),
]


def load_kb(filepath: str) -> list[dict]:
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def refine_age_range(text: str) -> str:
    """更精细的年龄段检测"""
    scores = {}
    for age, keywords in AGE_KEYWORDS_V2.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[age] = score
    if not scores:
        return "通用"
    return max(scores, key=scores.get)


def extract_entities(text: str) -> dict[str, list[str]]:
    """从文本中提取实体"""
    found = {}
    for entity_type, keywords in ENTITY_PATTERNS.items():
        hits = [kw for kw in keywords if kw in text]
        if hits:
            found[entity_type] = hits
    return found


def score_quality(record: dict) -> float:
    """综合质量评分 (0-1)"""
    content = record.get("content", "")
    title = record.get("title", "")
    source = record.get("source", "")

    score = 0.0

    # 长度评分
    clen = len(content)
    if clen > 2000:
        score += 0.3
    elif clen > 800:
        score += 0.2
    elif clen > 300:
        score += 0.1

    # 中文比例
    cn = sum(1 for c in content if '一' <= c <= '鿿')
    cn_ratio = cn / max(clen, 1)
    if cn_ratio > 0.8:
        score += 0.2
    elif cn_ratio > 0.5:
        score += 0.1

    # 结构化程度
    paragraphs = len(re.findall(r'\n\n|\n\s*\n', content))
    if paragraphs > 3:
        score += 0.15
    elif paragraphs > 1:
        score += 0.08

    # 是否有小标题或分点
    if re.search(r'[1-9][\.、）)]', content):
        score += 0.1

    # 来源权重
    source_weights = {
        "msd_manual": 1.0, "dxy": 1.0, "guideline": 1.0,
        "wikipedia_zh": 0.9, "wikipedia_en": 0.85, "wikipedia": 0.85,
        "zhihu": 0.7, "babytree": 0.65, "mama_cn": 0.65,
        "r1_distill": 0.5, "unknown": 0.3,
    }
    score += source_weights.get(source, 0.3) * 0.25

    return round(min(score, 1.0), 3)


def find_knowledge_gaps(records: list[dict]) -> list[dict]:
    """知识缺口分析"""
    # 统计当前覆盖
    coverage = defaultdict(int)
    for r in records:
        key = f"{r.get('category', '其他')}|{r.get('age_range', '通用')}"
        coverage[key] += 1

    gaps = []
    for category, age_range, threshold in COVERAGE_CHECKS:
        key = f"{category}|{age_range}"
        current = coverage.get(key, 0)
        if current < threshold:
            gaps.append({
                "category": category,
                "age_range": age_range,
                "current": current,
                "threshold": threshold,
                "shortfall": threshold - current,
            })

    gaps.sort(key=lambda g: g["shortfall"], reverse=True)
    return gaps


def enhance_records(records: list[dict]) -> list[dict]:
    """增强所有记录"""
    enhanced = []
    for i, r in enumerate(records):
        title = r.get("title", "")
        content = r.get("content", "")

        # 精细化年龄段
        refined_age = refine_age_range(title + content)
        if refined_age != "通用" and r.get("age_range", "通用") == "通用":
            r["age_range"] = refined_age

        # 实体提取
        entities = extract_entities(title + content)
        if entities:
            r["entities"] = entities

        # 质量评分
        r["quality_score"] = score_quality(r)

        enhanced.append(r)

        if (i + 1) % 1000 == 0:
            print(f"  Enhanced {i+1}/{len(records)}")

    return enhanced


def generate_report(records: list[dict], gaps: list[dict]) -> dict:
    """生成质量报告"""
    # 类别分布
    categories = Counter(r.get("category", "其他") for r in records)
    age_ranges = Counter(r.get("age_range", "通用") for r in records)
    sources = Counter(r.get("source", "unknown") for r in records)

    # 质量分布
    scores = [r.get("quality_score", 0) for r in records]
    high = sum(1 for s in scores if s >= 0.7)
    medium = sum(1 for s in scores if 0.4 <= s < 0.7)
    low = sum(1 for s in scores if s < 0.4)

    # 实体统计
    entity_counts = defaultdict(int)
    for r in records:
        for etype in r.get("entities", {}):
            entity_counts[etype] += len(r["entities"][etype])

    report = {
        "total_records": len(records),
        "quality_distribution": {
            "high": high, "medium": medium, "low": low,
            "avg_score": round(sum(scores) / max(len(scores), 1), 3),
        },
        "category_distribution": dict(categories.most_common()),
        "age_range_distribution": dict(age_ranges.most_common()),
        "source_distribution": dict(sources.most_common()),
        "entity_statistics": dict(entity_counts),
        "knowledge_gaps": gaps[:20],
        "gap_summary": {
            "total_gaps": len(gaps),
            "critical_gaps": len([g for g in gaps if g["shortfall"] >= 10]),
            "moderate_gaps": len([g for g in gaps if 5 <= g["shortfall"] < 10]),
        },
    }
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="知识库质量增强")
    parser.add_argument("--report-only", action="store_true", help="仅生成报告")
    parser.add_argument("--gap-analysis", action="store_true", help="仅知识缺口分析")
    args = parser.parse_args()

    print("=" * 60)
    print("  知识库质量增强与评估")
    print("=" * 60)

    # 加载
    records = load_kb(KB_FILE)
    print(f"\n[STEP 1] Loaded {len(records)} records")

    if len(records) == 0:
        print("[ERROR] Knowledge base is empty")
        return

    # 缺口分析
    print("\n[STEP 2] Knowledge gap analysis...")
    gaps = find_knowledge_gaps(records)
    for g in gaps[:10]:
        print(f"  {g['category']}/{g['age_range']}: {g['current']} articles "
              f"(need {g['threshold']}, shortfall {g['shortfall']})")

    if args.gap_analysis:
        # 保存缺口报告
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(gaps, f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] Gap report saved to {REPORT_FILE}")
        return

    # 生成报告
    print("\n[STEP 3] Generating quality report...")
    report = generate_report(records, gaps)

    print(f"  Quality: {report['quality_distribution']}")
    print(f"  Top categories: {list(report['category_distribution'].items())[:5]}")
    print(f"  Top age ranges: {list(report['age_range_distribution'].items())[:5]}")
    print(f"  Critical gaps: {report['gap_summary']['critical_gaps']}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved to {REPORT_FILE}")

    if args.report_only:
        return

    # 增强
    print(f"\n[STEP 4] Enhancing records...")
    enhanced = enhance_records(records)

    # 保存增强版
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in enhanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(enhanced)} enhanced records to {OUTPUT_FILE}")

    print(f"\n[DONE] Enhancement complete")


if __name__ == "__main__":
    main()
