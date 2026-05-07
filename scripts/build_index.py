"""
构建向量索引（BGE Embedding + ChromaDB）

使用 BAAI/bge-small-zh-v1.5 生成 512 维语义向量，存入 ChromaDB。
同时保留 TF-IDF 作为 fallback。
"""

import json
import os
import re
import sys
import time
import numpy as np

# 修复 Windows GBK 终端编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "parenting_knowledge_base.jsonl")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 64


def jieba_tokenize(text: str) -> str:
    try:
        import jieba
        return " ".join(jieba.cut(text))
    except ImportError:
        result = []
        for char in text:
            if '一' <= char <= '鿿':
                result.append(f" {char} ")
            elif char.isalnum():
                result.append(char)
            else:
                result.append(" ")
        return " ".join("".join(result).split())


def load_knowledge_base(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def chunk_text(text: str, max_length: int = 800, overlap: int = 100) -> list[str]:
    """BGE 模型 512 token 限制，中文约 1 char/token，留余量"""
    if len(text) <= max_length:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[。.!?！？])\s*', text)
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) > max_length and current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + sent
        else:
            current += sent

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def build_embedding_index(records: list[dict], model, collection) -> int:
    """用 BGE 模型生成 embedding 并存入 ChromaDB"""

    # 准备文档
    docs_for_embedding = []
    doc_metas = []
    doc_ids = []

    for rec in records:
        content = rec.get("content", "")
        title = rec.get("title", "")
        source = rec.get("source", "")
        category = rec.get("category", "其他")
        age_range = rec.get("age_range", "通用")
        url = rec.get("url", "")
        rec_id = rec.get("id", "")

        enriched = f"标题：{title}。分类：{category}。适用年龄段：{age_range}。内容：{content}"
        chunks = chunk_text(enriched)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{rec_id}_{i}"
            docs_for_embedding.append(chunk)
            doc_metas.append({
                "title": title,
                "source": source,
                "category": category,
                "age_range": age_range,
                "url": url,
                "chunk_index": i,
                "doc_id": rec_id,
            })
            doc_ids.append(chunk_id)

    print(f"[INFO] Generating embeddings for {len(docs_for_embedding)} chunks...")

    # 批量生成 embedding（BGE 模型在 query 前加 "为这个句子生成表示以用于检索相关文章：" 前缀效果更好）
    all_embeddings = []
    for i in range(0, len(docs_for_embedding), BATCH_SIZE):
        batch = docs_for_embedding[i:i + BATCH_SIZE]
        # BGE 模型：passage 不需要前缀
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
        if (i + BATCH_SIZE) % 1000 == 0 or i + BATCH_SIZE >= len(docs_for_embedding):
            print(f"  [EMBED] {min(i + BATCH_SIZE, len(docs_for_embedding))}/{len(docs_for_embedding)}")

    all_embeddings = np.vstack(all_embeddings)

    # 分批写入 ChromaDB
    print(f"[INFO] Writing to ChromaDB...")
    for i in range(0, len(docs_for_embedding), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(docs_for_embedding))
        collection.add(
            ids=doc_ids[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            documents=docs_for_embedding[i:end],
            metadatas=doc_metas[i:end],
        )

    print(f"[INFO] ChromaDB indexed: {collection.count()} documents")
    return collection.count()


def build_tfidf_fallback(records: list[dict]):
    """构建 TF-IDF 索引作为 fallback"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle

    docs = []
    for rec in records:
        content = rec.get("content", "")
        title = rec.get("title", "")
        source = rec.get("source", "")
        category = rec.get("category", "其他")
        age_range = rec.get("age_range", "通用")

        enriched = f"标题：{title}。分类：{category}。适用年龄段：{age_range}。内容：{content}"
        chunks = chunk_text(enriched, max_length=1500, overlap=200)

        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{rec.get('id', '')}_{i}",
                "text": chunk,
                "metadata": {
                    "title": title, "source": source,
                    "category": category, "age_range": age_range,
                    "url": rec.get("url", ""),
                }
            })

    texts = [jieba_tokenize(d["text"]) for d in docs]
    vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True)
    embeddings = vectorizer.fit_transform(texts).toarray()

    np.savez_compressed(os.path.join(INDEX_DIR, "tfidf_index.npz"), embeddings=embeddings)
    with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[INFO] TF-IDF fallback: {len(docs)} docs, {embeddings.shape[1]} features")


def test_embedding_retrieval(model, collection):
    """测试嵌入检索效果"""
    print("\n" + "=" * 60)
    print("  检索测试（Embedding 语义检索）")
    print("=" * 60)

    queries = [
        "宝宝发烧怎么办",
        "母乳喂养的好处",
        "新生儿护理注意事项",
        "孕妇饮食要注意什么",
        "儿童疫苗什么时候打",
        "孩子晚上不睡觉怎么办",
        "如何给宝宝添加辅食",
    ]

    for q in queries:
        # BGE query 需要加前缀
        q_embedding = model.encode(
            q, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\n  Q: {q}")
        if results["ids"] and results["ids"][0]:
            for rank, (doc_id, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
                sim = 1 - dist  # cosine distance → similarity
                meta = results["metadatas"][0][rank] if results["metadatas"] else {}
                doc = results["documents"][0][rank] if results["documents"] else ""
                print(f"  [{rank + 1}] sim={sim:.4f} | {meta.get('title', '')[:50]}")
                print(f"      {doc[:120]}...")
        else:
            print(f"  [NO RESULTS]")


def build_embedding_index_incremental(records: list[dict], model, collection) -> int:
    """增量模式：只编码新 chunk 追加到现有 ChromaDB"""
    existing_ids = set()
    try:
        existing = collection.get(include=[])
        if existing is not None and isinstance(existing, dict):
            ids_val = existing.get("ids")
            if ids_val is not None:
                existing_ids = set(ids_val)
    except Exception:
        pass

    print(f"[INFO] Existing chunks: {len(existing_ids)}")

    docs_for_embedding = []
    doc_metas = []
    doc_ids = []

    for rec in records:
        content = rec.get("content", "")
        title = rec.get("title", "")
        source = rec.get("source", "")
        category = rec.get("category", "其他")
        age_range = rec.get("age_range", "通用")
        url = rec.get("url", "")
        rec_id = rec.get("id", "")

        enriched = f"标题：{title}。分类：{category}。适用年龄段：{age_range}。内容：{content}"
        chunks = chunk_text(enriched)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{rec_id}_{i}"
            if chunk_id in existing_ids:
                continue  # 跳过已存在的 chunk
            docs_for_embedding.append(chunk)
            doc_metas.append({
                "title": title, "source": source,
                "category": category, "age_range": age_range,
                "url": url, "chunk_index": i, "doc_id": rec_id,
            })
            doc_ids.append(chunk_id)

    if not docs_for_embedding:
        print("[INFO] No new chunks to add.")
        return collection.count()

    print(f"[INFO] New chunks to embed: {len(docs_for_embedding)}")

    all_embeddings = []
    for i in range(0, len(docs_for_embedding), BATCH_SIZE):
        batch = docs_for_embedding[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
        if (i + BATCH_SIZE) % 1000 == 0 or i + BATCH_SIZE >= len(docs_for_embedding):
            print(f"  [EMBED] {min(i + BATCH_SIZE, len(docs_for_embedding))}/{len(docs_for_embedding)}")

    all_embeddings = np.vstack(all_embeddings)

    print(f"[INFO] Adding to ChromaDB...")
    for i in range(0, len(docs_for_embedding), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(docs_for_embedding))
        collection.add(
            ids=doc_ids[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            documents=docs_for_embedding[i:end],
            metadatas=doc_metas[i:end],
        )

    print(f"[INFO] ChromaDB now: {collection.count()} documents (+{len(docs_for_embedding)})")
    return collection.count()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--incremental", action="store_true", help="增量模式：只添加新数据")
    args = parser.parse_args()

    os.makedirs(INDEX_DIR, exist_ok=True)

    # 加载知识库
    print("[STEP 1] Loading knowledge base...")
    records = load_knowledge_base(KB_FILE)
    print(f"  Loaded {len(records)} records")

    if len(records) == 0:
        print("[ERROR] No records in knowledge base. Run process_knowledge.py first.")
        return

    # 加载 BGE 模型
    print(f"\n[STEP 2] Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded in {time.time() - t0:.0f}s, dim={model.get_sentence_embedding_dimension()}")

    # 构建 ChromaDB 嵌入索引
    import chromadb
    client = chromadb.PersistentClient(path=INDEX_DIR)

    if args.incremental:
        print(f"\n[STEP 3] Incremental index update...")
        try:
            collection = client.get_collection("parenting_kb")
        except Exception:
            print("[INFO] No existing collection, creating new one...")
            collection = client.create_collection(
                name="parenting_kb", metadata={"hnsw:space": "cosine"},
            )
        doc_count = build_embedding_index_incremental(records, model, collection)
    else:
        print(f"\n[STEP 3] Building ChromaDB embedding index (full rebuild)...")
        try:
            client.delete_collection("parenting_kb")
        except Exception:
            pass
        collection = client.create_collection(
            name="parenting_kb", metadata={"hnsw:space": "cosine"},
        )
        doc_count = build_embedding_index(records, model, collection)

    print(f"  ChromaDB: {doc_count} chunks indexed")

    # 构建 TF-IDF fallback
    print(f"\n[STEP 4] Building TF-IDF fallback...")
    build_tfidf_fallback(records)

    # 测试检索
    print(f"\n[STEP 5] Testing retrieval...")
    test_embedding_retrieval(model, collection)

    print(f"\n[DONE] Index: {INDEX_DIR}")
    print(f"[DONE] Embedding dim: {model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()
