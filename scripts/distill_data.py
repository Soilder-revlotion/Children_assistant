"""
RAG 增强蒸馏 — 基于已有知识库生成高质量 Q&A 对和变体

策略:
  1. 从知识库中采样不同类别/年龄段的文章
  2. 用 Ollama Qwen2.5:3b 生成 Q&A 对、年龄段适配版、实操建议
  3. 质量校验后追加到知识库

用法:
  python scripts/distill_data.py                # 默认蒸馏 200 条
  python scripts/distill_data.py --samples 100  # 采样 100 条蒸馏
  python scripts/distill_data.py --dry-run      # 预览但不保存
"""

import json
import os
import re
import sys
import time
import random
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base",
                       "parenting_knowledge_base.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                           "r1_distill_parenting.jsonl")
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"
MAX_TOKENS = 600
TEMPERATURE = 0.8


def load_kb(filepath: str) -> list[dict]:
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def sample_diverse(records: list[dict], n: int) -> list[dict]:
    """按类别和年龄段分层采样"""
    buckets = {}
    for r in records:
        key = f"{r.get('category', '其他')}|{r.get('age_range', '通用')}"
        buckets.setdefault(key, []).append(r)

    sampled = []
    # 每桶均匀采样
    per_bucket = max(1, n // max(len(buckets), 1))
    for bucket_records in buckets.values():
        random.shuffle(bucket_records)
        sampled.extend(bucket_records[:per_bucket])

    random.shuffle(sampled)
    return sampled[:n]


def generate_qa_pair(client: httpx.Client, record: dict) -> list[KnowledgeItem]:
    """根据文章生成 Q&A 对"""
    title = record.get("title", "")
    content = record.get("content", "")[:800]
    category = record.get("category", "")
    age_range = record.get("age_range", "")

    prompt = f"""你是一个专业的育儿知识编辑。根据下面的育儿文章，生成 2 个常见家长问题和对应回答。

文章标题：{title}
所属分类：{category}
适用年龄段：{age_range}
文章内容：
{content}

请生成 2 个 Q&A 对，格式如下（严格按格式）：
Q: 问题1
A: 回答1

Q: 问题2
A: 回答2

要求：
- 问题要真实、常见，是家长真正会问的
- 回答要基于文章内容，准确、简洁（100-200字）
- 回答要有实质内容，不要空泛的安慰话
- 涉及医疗时，回答末尾加一句"以上内容仅供参考，请咨询专业医生。" """

    try:
        resp = client.post("/chat/completions", json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }, timeout=60)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        # 解析 Q&A 对
        items = []
        qa_pairs = re.split(r"\n(?=Q:\s)", text)
        for qa in qa_pairs:
            qm = re.match(r"Q:\s*(.+?)(?:\n|$)", qa)
            am = re.search(r"A:\s*(.+?)(?:\n*(?:Q:|$))", qa, re.DOTALL)
            if not qm:
                continue
            question = qm.group(1).strip()
            answer = am.group(1).strip() if am else qa[qa.find("\n") + 1:].strip()
            answer = re.sub(r"^A:\s*", "", answer)

            if len(question) < 5 or len(answer) < 50:
                continue
            if chinese_char_ratio(question + answer) < 0.5:
                continue

            items.append(KnowledgeItem(
                source="r1_distill",
                type="qa",
                title=question,
                question=question,
                content=answer,
                age_range=guess_age_range(question + answer),
                category=guess_category(question + answer),
            ))

        return items

    except Exception as e:
        print(f"  [DISTILL ERROR] {e}")
        return []


def generate_age_variant(client: httpx.Client, record: dict, target_age: str) -> KnowledgeItem | None:
    """生成特定年龄段的适配版本"""
    title = record.get("title", "")
    content = record.get("content", "")[:600]

    prompt = f"""请将以下育儿知识改编成适合"{target_age}"阶段的版本。

原标题：{title}
原内容：
{content}

要求：
- 改编后标题和内容要针对{target_age}阶段的特点
- 保留原文的核心知识点
- 调整语言和细节使其与{target_age}阶段匹配
- 150-300字
- 直接输出改编后的内容，格式：标题在第一行，空一行后接正文"""

    try:
        resp = client.post("/chat/completions", json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }, timeout=60)

        text = resp.json()["choices"][0]["message"]["content"].strip()

        parts = text.split("\n\n", 1)
        new_title = parts[0].strip().lstrip("#").strip()
        new_content = parts[1].strip() if len(parts) > 1 else text

        if len(new_title) < 3 or len(new_content) < 100:
            return None
        if chinese_char_ratio(new_content) < 0.5:
            return None

        return KnowledgeItem(
            source="r1_distill",
            type="article",
            title=new_title,
            content=new_content,
            age_range=target_age,
            category=record.get("category", "其他"),
        )

    except Exception as e:
        print(f"  [AGE VARIANT ERROR] {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="育儿知识数据蒸馏")
    parser.add_argument("--samples", type=int, default=200,
                        help="采样文章数 (默认: 200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览不保存")
    parser.add_argument("--qa-only", action="store_true",
                        help="仅生成 Q&A 对")
    parser.add_argument("--age", type=str, default=None,
                        help="仅生成特定年龄段变体 (如: 0-1月,1-6月,6-12月)")
    args = parser.parse_args()

    print("=" * 60)
    print("  育儿知识数据增强蒸馏")
    print("=" * 60)

    # 加载知识库
    records = load_kb(KB_FILE)
    print(f"[KB] Loaded {len(records)} records")

    if len(records) == 0:
        print("[ERROR] Knowledge base is empty")
        return

    # 采样
    samples = sample_diverse(records, args.samples)
    print(f"[SAMPLE] Selected {len(samples)} diverse records")

    # 连接 Ollama
    print(f"[LLM] Connecting to {OLLAMA_URL} ({OLLAMA_MODEL})...")
    client = httpx.Client(base_url=OLLAMA_URL, timeout=60)

    # 测试连接
    try:
        r = client.get("/models")
        if r.status_code == 200:
            print(f"[LLM] Connected OK")
        else:
            print(f"[ERROR] Ollama not reachable: HTTP {r.status_code}")
            return
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama: {e}")
        print("[HINT] Make sure Ollama is running: ollama serve")
        return

    total_generated = 0

    if args.age:
        # 仅生成特定年龄段变体
        print(f"\n[DISTILL] Generating age variants for: {args.age}")
        for i, rec in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] {rec.get('title', '')[:50]}")
            variant = generate_age_variant(client, rec, args.age)
            if variant:
                if not args.dry_run:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(variant.__dict__, ensure_ascii=False) + "\n")
                total_generated += 1
                print(f"    [OK] {variant.title[:50]}")
            time.sleep(random.uniform(0.3, 0.8))
    else:
        # Q&A 对生成模式
        print(f"\n[DISTILL] Generating Q&A pairs from {len(samples)} samples...")
        for i, rec in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] {rec.get('title', '')[:50]}")

            qa_items = generate_qa_pair(client, rec)
            for item in qa_items:
                if is_valid_knowledge(item):
                    if not args.dry_run:
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                    total_generated += 1
                    print(f"    [QA] {item.question[:50]}")

            # 为每条记录也生成一个年龄段变体
            if not args.qa_only and random.random() < 0.3:
                target_ages = ["0-1月", "1-6月", "6-12月", "1-3岁"]
                age = random.choice(target_ages)
                variant = generate_age_variant(client, rec, age)
                if variant and is_valid_knowledge(variant):
                    if not args.dry_run:
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(variant.__dict__, ensure_ascii=False) + "\n")
                    total_generated += 1
                    print(f"    [AGE-{age}] {variant.title[:50]}")

            time.sleep(random.uniform(0.3, 0.8))

    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\n[DONE] Generated {total_generated} items{suffix}")
    if not args.dry_run and total_generated > 0:
        print(f"[DONE] Output: {OUTPUT_FILE}")
        if not args.qa_only and not args.age:
            print(f"[NEXT] Run process_knowledge.py → audit → build_index")


if __name__ == "__main__":
    main()
