"""
知识库质量自检脚本

用 BGE Embedding 对每条知识做质量审计：
1. 语义相似度 vs 育儿/医学术语锚点 → 相关性评分
2. 广告/营销内容检测
3. 语义去重（相似度 > 0.95）
4. 内容长度降权

用法:
  python scripts/audit_knowledge.py              # 审计并输出报告
  python scripts/audit_knowledge.py --clean      # 审计并生成清洗后知识库
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from collections import defaultdict

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "parenting_knowledge_base.jsonl")
CLEANED_KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "parenting_knowledge_base_cleaned.jsonl")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "audit_report.json")

# 育儿/医学锚点术语 — 用于计算语义相似度
PARENTING_ANCHORS = [
    "婴儿护理与喂养知识，包括母乳喂养、配方奶选择、辅食添加时间",
    "儿童常见疾病预防与治疗，如发烧、感冒、咳嗽、腹泻、湿疹的处理方法",
    "孕期保健与产后恢复，包括产检、营养补充、分娩准备",
    "儿童生长发育里程碑，包括身高体重标准、大运动发展、语言发育",
    "儿童疫苗接种时间表和注意事项",
    "儿童心理健康与行为引导，包括情绪管理、社交能力培养",
    "新生儿护理要点，包括脐带护理、黄疸观察、睡眠安全",
    "幼儿早教与启蒙教育方法",
    "儿童营养与饮食搭配，包括维生素补充、挑食偏食纠正",
    "儿科用药安全与急救知识",
]

# 广告/营销关键词
AD_KEYWORDS = [
    "加微信", "扫码", "免费领", "点击购买", "下单", "优惠券",
    "限时抢购", "团购", "拼团", "秒杀", "满减", "包邮",
    "添加好友", "私信", "关注公众号", "下载APP", "注册送",
    "代理", "加盟", "招商", "赚钱", "月入", "兼职",
    "全网最低", "正品保证", "无效退款", "立即咨询",
    "➕", "📱", "💬",
]

# 低质量内容特征
LOW_QUALITY_PATTERNS = [
    (r"小红书风格", "AI水军风格内容"),
    (r"写一篇.*帖子", "AI水军风格内容"),
    (r"写一篇.*回答", "AI水军风格内容"),
    (r"请.*写.*文章", "AI prompt 残留"),
    (r"请.*回答", "AI prompt 残留"),
    (r"作为.*AI", "AI 自我引用"),
    (r"作为.*语言模型", "AI 自我引用"),
]


def load_knowledge_base(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_anchor_embeddings(model, anchors: list[str]) -> np.ndarray:
    """为锚点术语生成 BGE embedding"""
    embeddings = model.encode(anchors, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


def compute_semantic_score(
    content: str, model, anchor_embeddings: np.ndarray
) -> float:
    """计算内容与育儿锚点的最高语义相似度"""
    if len(content) < 20:
        return 0.0
    try:
        # 取内容前 1000 字符（足够判断主题）
        truncated = content[:1000]
        content_emb = model.encode(
            [truncated], normalize_embeddings=True, show_progress_bar=False
        )
        similarities = np.dot(content_emb, anchor_embeddings.T).flatten()
        return float(np.max(similarities))
    except Exception:
        return 0.0


def detect_advertising(content: str, title: str) -> tuple[bool, list[str]]:
    """检测广告/营销内容"""
    text = title + " " + content
    detected = []
    for kw in AD_KEYWORDS:
        if kw in text:
            detected.append(kw)
    return len(detected) > 0, detected


def detect_low_quality(content: str, title: str) -> tuple[bool, list[str]]:
    """检测低质量/AI水军内容"""
    text = title + " " + content[:500]
    issues = []
    import re
    for pattern, label in LOW_QUALITY_PATTERNS:
        if re.search(pattern, text):
            issues.append(label)
    return len(issues) > 0, issues


def audit(records: list[dict], model) -> list[dict]:
    """对每条记录审计打分"""
    print(f"[INFO] Building anchor embeddings...")
    anchor_embeddings = build_anchor_embeddings(model, PARENTING_ANCHORS)

    print(f"[INFO] Auditing {len(records)} records...")
    audited = []
    stats = {
        "semantic_pass": 0,
        "semantic_warn": 0,
        "semantic_fail": 0,
        "ad_detected": 0,
        "low_quality": 0,
        "too_short": 0,
    }

    for i, rec in enumerate(records):
        content = rec.get("content", "")
        title = rec.get("title", "")
        text = title + " " + content

        # 语义评分
        semantic_score = compute_semantic_score(text, model, anchor_embeddings)

        # 质量等级
        if semantic_score >= 0.5:
            quality = "high"
            stats["semantic_pass"] += 1
        elif semantic_score >= 0.35:
            quality = "medium"
            stats["semantic_warn"] += 1
        else:
            quality = "low"
            stats["semantic_fail"] += 1

        # 广告检测
        is_ad, ad_kw = detect_advertising(content, title)
        if is_ad:
            stats["ad_detected"] += 1

        # 低质量检测
        is_low, low_issues = detect_low_quality(content, title)
        if is_low:
            stats["low_quality"] += 1

        # 长度检查
        if len(content) < 100:
            stats["too_short"] += 1

        # 综合决策
        should_remove = is_ad or is_low or (semantic_score < 0.3)
        should_downgrade = semantic_score < 0.45 or len(content) < 100

        audited.append({
            **rec,
            "audit_semantic_score": round(float(semantic_score), 4),
            "audit_quality": quality,
            "audit_is_ad": is_ad,
            "audit_low_quality": is_low,
            "audit_should_remove": should_remove,
            "audit_should_downgrade": should_downgrade,
        })

        if (i + 1) % 500 == 0:
            print(f"  [AUDIT] {i + 1}/{len(records)} | "
                  f"high={stats['semantic_pass']} "
                  f"med={stats['semantic_warn']} "
                  f"low={stats['semantic_fail']} "
                  f"ad={stats['ad_detected']} "
                  f"junk={stats['low_quality']}")

    print(f"\n[AUDIT DONE] {len(records)} records")
    print(f"  semantic high:   {stats['semantic_pass']}")
    print(f"  semantic medium: {stats['semantic_warn']}")
    print(f"  semantic low:    {stats['semantic_fail']}")
    print(f"  ad detected:     {stats['ad_detected']}")
    print(f"  low quality:     {stats['low_quality']}")
    print(f"  too short:       {stats['too_short']}")

    return audited, stats


def semantic_dedup(audited: list[dict], model, threshold: float = 0.95) -> list[dict]:
    """语义去重 — 对每条计算 embedding，相似度 > threshold 的标记为重复"""
    print(f"\n[INFO] Semantic deduplication (threshold={threshold})...")
    print(f"[INFO] Generating embeddings for {len(audited)} records...")

    texts = []
    for rec in audited:
        content = rec.get("content", "")
        texts.append(content[:800])  # 前 800 字足够判断

    print(f"[INFO] Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64,
    )

    # 分批计算相似度矩阵（避免 O(n²) 内存问题）
    removed_ids = set()
    dup_count = 0

    # 使用分块策略
    chunk_size = 500
    for i in range(0, len(texts), chunk_size):
        end_i = min(i + chunk_size, len(texts))
        chunk_emb = embeddings[i:end_i]

        for j in range(i, len(texts), chunk_size):
            if j in removed_ids:
                continue
            end_j = min(j + chunk_size, len(texts))
            other_emb = embeddings[j:end_j]

            sims = np.dot(chunk_emb, other_emb.T)

            for ci in range(sims.shape[0]):
                global_ci = i + ci
                if global_ci in removed_ids:
                    continue
                for cj in range(sims.shape[1]):
                    global_cj = j + cj
                    if global_cj <= global_ci or global_cj in removed_ids:
                        continue
                    if sims[ci][cj] > threshold:
                        # 保留 quality 更高的那条
                        q1 = audited[global_ci].get("audit_semantic_score", 0)
                        q2 = audited[global_cj].get("audit_semantic_score", 0)
                        loser = global_cj if q1 >= q2 else global_ci
                        removed_ids.add(loser)
                        dup_count += 1

        if (i + chunk_size) % 1000 == 0 or i + chunk_size >= len(texts):
            print(f"  [DEDUP] processed {min(i + chunk_size, len(texts))}/{len(texts)}, "
                  f"duplicates found: {dup_count}")

    print(f"  [DEDUP] {dup_count} duplicates identified, removing {len(removed_ids)} records")

    return [r for idx, r in enumerate(audited) if idx not in removed_ids]


def generate_report(audited: list[dict], stats: dict, after_dedup: int):
    """生成审计报告"""
    report = {
        "total_original": len(audited),
        "total_after_dedup": after_dedup,
        "stats": stats,
        "by_source": defaultdict(lambda: {"total": 0, "removed": 0, "downgraded": 0, "high": 0, "medium": 0, "low": 0}),
        "by_category": defaultdict(lambda: {"total": 0, "removed": 0}),
        "removed_samples": [],
        "downgraded_samples": [],
    }

    for rec in audited:
        src = rec.get("source", "unknown")
        cat = rec.get("category", "其他")
        report["by_source"][src]["total"] += 1
        report["by_category"][cat]["total"] += 1

        quality = rec.get("audit_quality", "low")
        report["by_source"][src][quality] += 1

        if rec.get("audit_should_remove"):
            report["by_source"][src]["removed"] += 1
            report["by_category"][cat]["removed"] += 1
            if len(report["removed_samples"]) < 30:
                report["removed_samples"].append({
                    "id": rec.get("id", ""),
                    "title": rec.get("title", "")[:100],
                    "source": src,
                    "reason": {
                        "semantic_score": rec.get("audit_semantic_score", 0),
                        "is_ad": rec.get("audit_is_ad", False),
                        "low_quality": rec.get("audit_low_quality", False),
                    }
                })

        if rec.get("audit_should_downgrade") and not rec.get("audit_should_remove"):
            report["by_source"][src]["downgraded"] += 1
            if len(report["downgraded_samples"]) < 20:
                report["downgraded_samples"].append({
                    "id": rec.get("id", ""),
                    "title": rec.get("title", "")[:100],
                    "source": src,
                    "semantic_score": rec.get("audit_semantic_score", 0),
                })

    # 转换 defaultdict 为普通 dict
    report["by_source"] = dict(report["by_source"])
    report["by_category"] = dict(report["by_category"])

    return report


def main():
    parser = argparse.ArgumentParser(description="知识库质量审计")
    parser.add_argument("--clean", action="store_true", help="生成清洗后知识库")
    args = parser.parse_args()

    print("[STEP 1] Loading knowledge base...")
    records = load_knowledge_base(KB_FILE)
    print(f"  Loaded {len(records)} records")

    if not records:
        print("[ERROR] No records found")
        return

    print("\n[STEP 2] Loading BGE model...")
    from sentence_transformers import SentenceTransformer
    model_name = "BAAI/bge-small-zh-v1.5"
    t0 = time.time()
    model = SentenceTransformer(model_name)
    print(f"  Model loaded in {time.time() - t0:.0f}s, dim={model.get_sentence_embedding_dimension()}")

    print(f"\n[STEP 3] Auditing...")
    audited, stats = audit(records, model)

    print(f"\n[STEP 4] Semantic deduplication...")
    deduped = semantic_dedup(audited, model, threshold=0.95)
    print(f"  After dedup: {len(deduped)} records (removed {len(audited) - len(deduped)})")

    print(f"\n[STEP 5] Generating report...")
    report = generate_report(audited, stats, len(deduped))

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved to {REPORT_FILE}")

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"  审计报告摘要")
    print(f"{'=' * 60}")
    print(f"  原始条目:    {report['total_original']}")
    print(f"  去重后:      {report['total_after_dedup']}")
    should_remove = sum(1 for r in audited if r.get("audit_should_remove"))
    print(f"  建议删除:    {should_remove} ({should_remove/len(audited)*100:.1f}%)")
    should_downgrade = sum(1 for r in audited
                           if r.get("audit_should_downgrade") and not r.get("audit_should_remove"))
    print(f"  建议降权:    {should_downgrade} ({should_downgrade/len(audited)*100:.1f}%)")

    print(f"\n  按来源质量:")
    for src, s in sorted(report["by_source"].items(), key=lambda x: -x[1]["total"]):
        total = s["total"]
        high_pct = s["high"] / total * 100 if total else 0
        print(f"    {src:25s} total={total:5d}  high={high_pct:5.1f}%  "
              f"removed={s['removed']}")

    if args.clean:
        print(f"\n[STEP 6] Saving cleaned knowledge base...")
        # 保留不删除的记录
        cleaned = [r for r in deduped if not r.get("audit_should_remove")]
        # 降权记录：降低 quality_score
        for r in cleaned:
            if r.get("audit_should_downgrade"):
                r["quality_score"] = round(r.get("quality_score", 0.5) * 0.5, 2)
            # 删除审计专用字段
            for k in list(r.keys()):
                if k.startswith("audit_"):
                    del r[k]

        with open(CLEANED_KB_FILE, "w", encoding="utf-8") as f:
            for r in cleaned:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(cleaned)} records to {CLEANED_KB_FILE}")
        print(f"  Copy this file over {KB_FILE} and run build_index.py --rebuild to apply.")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
