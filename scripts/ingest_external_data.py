"""
存量数据下载与转换脚本

从 HuggingFace 下载中文数据集，过滤育儿相关内容，转为统一 Schema。

数据源:
  1. Chinese-DeepSeek-R1-Distill-data-110k → 过滤育儿相关 QA
  2. (后续可扩展其他数据源)

用法:
  python scripts/ingest_external_data.py
  python scripts/ingest_external_data.py --dry-run   # 仅预览，不保存
"""

import json
import os
import sys
import argparse
import time

# 修复 Windows GBK 终端编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "r1_distill_parenting.jsonl")
SKIPPED_FILE = os.path.join(OUTPUT_DIR, "r1_distill_non_parenting_sample.jsonl")

# 育儿关键词加权评分（避免单关键词误判）
# 权重 3：强特异性育儿词，几乎不会出现在非育儿上下文中
CORE_PARENTING_KW = {
    "母乳", "配方奶", "辅食", "断奶", "厌奶", "喂奶", "哺乳",
    "新生儿", "早产儿", "月子", "坐月子", "产检", "胎教", "宫缩",
    "哄睡", "夜醒", "夜奶", "并觉", "睡眠倒退",
    "脐带", "换尿布", "红屁股", "淹脖子", "抚触",
    "安全座椅", "手足口", "百白破", "流脑",
    "吸奶器", "催奶", "回奶",
    "幼儿急疹", "婴儿湿疹", "小儿", "儿科",
    "蒙台梭利", "感统训练", "精细动作", "大运动",
    "分离焦虑", "如厕训练", "入园焦虑",
}
# 权重 2：育儿常用词，大部分时候指向育儿
STRONG_PARENTING_KW = {
    "宝宝", "婴儿", "幼儿", "育儿", "母婴", "亲子",
    "孕妇", "怀孕", "孕期", "产后", "分娩",
    "奶粉", "奶瓶", "胎动", "预产期",
    "疫苗", "接种", "预防针",
    "爬行", "出牙", "长牙", "翻身",
    "早教", "启蒙", "绘本",
    "痱子", "湿疹", "黄疸",
    "猛涨期", "厌奶期", "睡眠倒退",
    "幼儿园", "入园", "小学生",
}
# 权重 1：弱信号，需要与其他词组合
WEAK_PARENTING_KW = {
    "孩子", "小孩", "儿童", "娃", "家长", "父母", "妈妈", "爸爸",
    "发育", "发烧", "咳嗽", "腹泻", "便秘", "过敏",
    "睡眠", "睡觉", "身高", "体重", "头围",
    "维生素D", "DHA", "益生菌", "鱼肝油",
    "补钙", "缺钙", "补铁", "缺铁", "补锌", "缺锌",
    "钙剂", "铁剂", "锌剂",
    "感冒", "肺炎", "水痘",
    "安全感", "情绪", "社交",
    "防撞", "防摔", "床围",
    "学步", "学说话", "认字",
}

# 硬排除词
EXCLUDE_KEYWORDS = [
    "数学", "证明", "算法", "编程", "代码", "服务器",
    "投资", "股票", "基金", "理财", "比特币",
    "汽车", "驾照", "购房", "装修", "家具",
    "化学式", "方程式", "微积分", "几何", "物理",
    "C++", "Python", "Java", "JavaScript", "SQL",
    "区块链", "合约", "NFT", "元宇宙",
    "免费送", "福利", "抽奖", "薅羊毛",
    "数学题", "应用题",
    "新冠", "新冠肺炎", "COVID",
    "流星雨", "天文",
    "模式生物", "实验动物",
]


def is_parenting_related(text: str) -> bool:
    """加权评分判断育儿相关性。总分 >= 3 视为育儿相关。"""
    text_lower = text.lower()

    # 硬排除
    for kw in EXCLUDE_KEYWORDS:
        if kw in text_lower:
            return False

    score = 0
    for kw in CORE_PARENTING_KW:
        if kw in text_lower:
            score += 3
    for kw in STRONG_PARENTING_KW:
        if kw in text_lower:
            score += 2
    for kw in WEAK_PARENTING_KW:
        if kw in text_lower:
            score += 1

    return score >= 3


# 本地缓存的 JSONL 文件路径
CACHED_JSONL = os.path.join(
    os.path.expanduser("~"),
    ".cache", "modelscope", "hub", "datasets", "downloads",
    "161f355c6715c73944d91767cfbf168fe6d61e04ea8314697622e139dfa68186",
)


def process_r1_dataset(dry_run: bool = False):
    """处理 R1 蒸馏数据集（从本地缓存的 JSONL 直接读取，无需网络）"""
    if not os.path.exists(CACHED_JSONL):
        print(f"[ERROR] Cached JSONL not found: {CACHED_JSONL}")
        print("[INFO] Attempting ModelScope download...")
        from modelscope.msdatasets import MsDataset
        MsDataset.load(
            "liucong/Chinese-DeepSeek-R1-Distill-data-110k",
            subset_name="default",
            split="train",
        )
        if not os.path.exists(CACHED_JSONL):
            print("[ERROR] Download failed. Skipping R1 dataset.")
            return [], []

    file_size_mb = os.path.getsize(CACHED_JSONL) / (1024 * 1024)
    print(f"[INFO] Reading local JSONL: {CACHED_JSONL}")
    print(f"[INFO] File size: {file_size_mb:.0f} MB")

    total = 0
    parenting_count = 0
    items = []
    skipped_samples = []

    t_start = time.time()

    with open(CACHED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = (sample.get("input") or "").strip()
            answer = (sample.get("content") or "").strip()
            score = sample.get("score", 0)
            repo = sample.get("repo_name", "")

            if not question or not answer:
                continue

            # 用问题 + 答案的前 500 字做育儿判断
            text_to_check = question + " " + answer[:500]

            if not is_parenting_related(text_to_check):
                if len(skipped_samples) < 20:
                    skipped_samples.append({
                        "question": question[:100],
                        "answer": answer[:100],
                        "repo": repo,
                    })
                continue

            parenting_count += 1

            # 构建 KnowledgeItem
            item = KnowledgeItem(
                source="r1_distill",
                type="qa",
                title=question[:200],
                question=question,
                content=answer,
                age_range=guess_age_range(question + answer[:500]),
                category=guess_category(question + answer[:500]),
                quality_score=float(score) / 10.0 if score else 0.5,
                url="",
            )

            if is_valid_knowledge(item):
                items.append(item.__dict__)

            # 进度输出
            if total % 5000 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  [PROGRESS] scanned={total}, found={parenting_count}, "
                      f"rate={rate:.0f}/s, elapsed={elapsed:.0f}s")

            # dry-run 模式：找到 10 条就停止
            if dry_run and len(items) >= 10:
                print(f"  [DRY-RUN] Stopping after {len(items)} samples")
                break

    elapsed = time.time() - t_start
    print(f"\n[DONE] Scanned: {total}, Parenting: {parenting_count} "
          f"({parenting_count/total*100:.1f}%)" if total else "")
    print(f"[DONE] Time: {elapsed:.0f}s")

    return items, skipped_samples


def main():
    parser = argparse.ArgumentParser(description="下载并过滤外部数据集")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不保存")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理 R1 蒸馏数据集
    items, skipped = process_r1_dataset(dry_run=args.dry_run)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("  预览 - 找到的育儿 QA 样本")
        print("=" * 60)
        for i, item in enumerate(items):
            print(f"\n[{i+1}] {item['title'][:80]}")
            print(f"    age={item['age_range']}, cat={item['category']}, "
                  f"score={item['quality_score']}")
            print(f"    content: {item['content'][:150]}...")

        if skipped:
            print("\n" + "=" * 60)
            print("  被过滤的样本（非育儿）")
            print("=" * 60)
            for i, s in enumerate(skipped[:10]):
                print(f"\n[{i+1}] {s['question'][:80]}")
                print(f"    {s['answer'][:100]}...")
        return

    if items:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[DONE] Saved {len(items)} parenting QA pairs to {OUTPUT_FILE}")

        # 统计
        from collections import Counter
        ages = Counter(i["age_range"] for i in items)
        cats = Counter(i["category"] for i in items)
        print(f"\n  年龄段分布: {ages.most_common()}")
        print(f"  分类分布: {cats.most_common(10)}")
    else:
        print(f"\n[WARN] No parenting items found in the dataset.")


if __name__ == "__main__":
    main()
