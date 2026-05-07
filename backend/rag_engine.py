"""RAG 引擎：BGE Embedding 语义检索 + Reranker 精排 + HyDE 增强 + 来源权重 + LLM 生成"""

import json
import os
import pickle
import time
from typing import Optional

import numpy as np

from backend.config import (
    CHROMADB_PATH, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD,
    LLM_CONFIGS, LLM_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIM,
    RERANKER_MODEL, USE_RERANKER, RERANKER_TOP_K, RERANKER_FINAL_K,
    USE_HYDE, HYDE_MAX_TOKENS, HYDE_PROMPT,
    SOURCE_WEIGHTS,
    RAG_SYSTEM_PROMPT, RAG_USER_PROMPT, FALLBACK_RESPONSE,
)


class RAGEngine:
    def __init__(self):
        self.llm_config = LLM_CONFIGS.get(LLM_PROVIDER, LLM_CONFIGS["ollama"])
        self.collection = None
        self.embedding_model = None
        self.reranker_model = None
        self.tfidf_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_docs = None
        self.use_chromadb = False
        self.use_embedding = False
        self._init()

    def _init(self):
        os.makedirs(CHROMADB_PATH, exist_ok=True)

        # BGE Embedding (可通过 SKIP_BGE=1 跳过，回退到 ChromaDB 内置 embedding)
        if os.getenv("SKIP_BGE", "").lower() in ("1", "true", "yes"):
            print("[INFO] SKIP_BGE=1, using ChromaDB built-in embedding")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                self.use_embedding = True
                print(f"[INFO] BGE embedding model loaded, dim={self.embedding_model.get_sentence_embedding_dimension()}")
            except Exception as e:
                print(f"[WARN] BGE model not available: {e}")

        # Reranker (lazy load on first use to save startup time)
        if USE_RERANKER:
            print(f"[INFO] Reranker configured: {RERANKER_MODEL} (lazy load)")

        # ChromaDB
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMADB_PATH)
            self.collection = client.get_collection("parenting_kb")
            self.use_chromadb = True
            print(f"[INFO] ChromaDB connected, {self.collection.count()} docs")
            return
        except Exception as e:
            print(f"[INFO] ChromaDB not available: {e}")

        # Fallback: TF-IDF
        tfidf_path = os.path.join(CHROMADB_PATH, "tfidf_index.npz")
        docs_path = os.path.join(CHROMADB_PATH, "docs.json")
        vec_path = os.path.join(CHROMADB_PATH, "vectorizer.pkl")

        if os.path.exists(tfidf_path) and os.path.exists(docs_path):
            try:
                data = np.load(tfidf_path)
                self.tfidf_embeddings = data["embeddings"]
                with open(docs_path, "r", encoding="utf-8") as f:
                    self.tfidf_docs = json.load(f)
                if os.path.exists(vec_path):
                    with open(vec_path, "rb") as f:
                        self.tfidf_vectorizer = pickle.load(f)
                print(f"[INFO] TF-IDF index loaded, {len(self.tfidf_docs)} docs")
                return
            except Exception as e:
                print(f"[ERROR] Failed to load TF-IDF index: {e}")

        print("[WARN] No index available. Run build_index.py first.")

    # ========== Reranker ==========

    def _load_reranker(self):
        """Lazy load reranker model"""
        if self.reranker_model is not None:
            return
        if not USE_RERANKER:
            return
        try:
            from sentence_transformers import CrossEncoder
            print(f"[INFO] Loading reranker: {RERANKER_MODEL}...")
            t0 = time.time()
            self.reranker_model = CrossEncoder(RERANKER_MODEL)
            print(f"[INFO] Reranker loaded in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"[WARN] Reranker not available: {e}")

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """用 CrossEncoder 精排检索结果"""
        if not docs or not USE_RERANKER:
            return docs[:RERANKER_FINAL_K]

        self._load_reranker()
        if self.reranker_model is None:
            return docs[:RERANKER_FINAL_K]

        pairs = []
        for doc in docs:
            content = doc.get("content", "")
            title = doc.get("title", "")
            text = f"{title} {content}"[:1500]
            pairs.append([query, text])

        try:
            scores = self.reranker_model.predict(pairs, show_progress_bar=False)
            for i, doc in enumerate(docs):
                doc["rerank_score"] = round(float(scores[i]), 4)
            docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")

        return docs[:RERANKER_FINAL_K]

    # ========== HyDE ==========

    def _generate_hypothetical(self, question: str) -> str:
        """生成假设回答用于检索增强"""
        api_key = self.llm_config.get("api_key", "")
        if not api_key:
            return ""

        prompt = HYDE_PROMPT.format(question=question)
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.llm_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": HYDE_MAX_TOKENS,
                "stream": False,
            }
            resp = httpx.post(
                f"{self.llm_config['base_url']}/chat/completions",
                json=body, headers=headers, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] HyDE generation failed: {e}")
            return ""

    def _merge_dedup_results(
        self, results_a: list[dict], results_b: list[dict]
    ) -> list[dict]:
        """合并两次检索结果，按相似度去重"""
        seen_ids = set()
        merged = []
        for doc in results_a:
            doc_id = doc.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)
        for doc in results_b:
            doc_id = doc.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # 标记来自 HyDE
                doc["via_hyde"] = True
                merged.append(doc)
        return merged

    # ========== 来源权重 ==========

    def _apply_source_weights(self, docs: list[dict]) -> list[dict]:
        """根据数据金字塔来源加权调整相似度"""
        for doc in docs:
            source = doc.get("source", "")
            weight = SOURCE_WEIGHTS.get(source, 0.5)
            original_sim = doc.get("similarity", 0)
            # 加权：低质量源降权，高质量源不变
            doc["similarity_raw"] = original_sim
            doc["similarity"] = round(original_sim * weight, 4)
            doc["source_weight"] = weight

        # 按加权相似度重新排序
        docs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return docs

    # ========== 检索 ==========

    def retrieve(
        self, query: str, top_k: int = TOP_K_RETRIEVAL,
        use_hyde: bool = None, use_reranker: bool = None,
    ) -> list[dict]:
        """语义检索 + 可选 HyDE 增强 + Reranker 精排 + 来源权重"""

        if use_hyde is None:
            use_hyde = USE_HYDE
        if use_reranker is None:
            use_reranker = USE_RERANKER

        # Step 1: 直接查询检索
        docs = self._do_search(query, top_k=top_k if not use_reranker else RERANKER_TOP_K)

        # Step 2: HyDE 增强（可选）
        if use_hyde and docs:
            hypothetical = self._generate_hypothetical(query)
            if hypothetical and len(hypothetical) > 5:
                hyde_docs = self._do_search(hypothetical, top_k=top_k if not use_reranker else RERANKER_TOP_K)
                docs = self._merge_dedup_results(docs, hyde_docs)

        # Step 3: Reranker 精排（可选）
        if use_reranker and len(docs) > RERANKER_FINAL_K:
            docs = self._rerank(query, docs)

        # Step 4: 来源权重
        docs = self._apply_source_weights(docs)

        # Step 5: 截断 + 阈值过滤
        docs = docs[:top_k]
        docs = [d for d in docs if d.get("similarity", 0) >= SIMILARITY_THRESHOLD]

        return docs

    def _do_search(self, query: str, top_k: int) -> list[dict]:
        """底层检索：Embedding → ChromaDB / TF-IDF"""
        if self.use_chromadb and self.use_embedding:
            return self._retrieve_embedding(query, top_k)
        elif self.use_chromadb:
            return self._retrieve_chromadb_text(query, top_k)
        elif self.tfidf_embeddings is not None:
            return self._retrieve_tfidf(query, top_k)
        else:
            return []

    def _retrieve_embedding(self, query: str, top_k: int) -> list[dict]:
        """BGE Embedding 语义检索"""
        try:
            query_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    docs.append({
                        "id": doc_id,
                        "content": document,
                        "similarity": round(similarity, 4),
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "category": metadata.get("category", ""),
                        "age_range": metadata.get("age_range", ""),
                        "url": metadata.get("url", ""),
                    })
            return docs
        except Exception as e:
            print(f"[ERROR] Embedding retrieval failed: {e}")
            return []

    def _retrieve_chromadb_text(self, query: str, top_k: int) -> list[dict]:
        """ChromaDB 内置 embedding 检索"""
        try:
            results = self.collection.query(
                query_texts=[query], n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance
                    if similarity < SIMILARITY_THRESHOLD:
                        continue
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    docs.append({
                        "id": doc_id, "content": document,
                        "similarity": round(similarity, 3),
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "category": metadata.get("category", ""),
                        "age_range": metadata.get("age_range", ""),
                        "url": metadata.get("url", ""),
                    })
            return docs
        except Exception as e:
            print(f"[ERROR] ChromaDB text retrieval failed: {e}")
            return []

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[dict]:
        """TF-IDF 关键词检索（fallback）"""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.build_index import jieba_tokenize

        if self.tfidf_vectorizer is not None:
            query_tokenized = jieba_tokenize(query)
            query_vec = self.tfidf_vectorizer.transform([query_tokenized]).toarray()
            dot_products = np.dot(self.tfidf_embeddings, query_vec.T).flatten()
            doc_norms = np.linalg.norm(self.tfidf_embeddings, axis=1)
            query_norm = np.linalg.norm(query_vec)
            norms = doc_norms * query_norm
            norms[norms == 0] = 1
            similarities = dot_products / norms
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            scores = np.array([
                sum(1 for w in query if w in doc.get("text", ""))
                for doc in self.tfidf_docs
            ], dtype=float)
            top_indices = np.argsort(scores)[::-1][:top_k]

        docs = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.01:
                continue
            doc = self.tfidf_docs[idx]
            metadata = doc.get("metadata", {})
            docs.append({
                "id": doc.get("id", str(idx)),
                "content": doc.get("text", ""),
                "similarity": round(sim, 4),
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "category": metadata.get("category", ""),
                "age_range": metadata.get("age_range", ""),
                "url": metadata.get("url", ""),
            })
        return docs

    # ========== 上下文构建 & LLM 生成 ==========

    def build_context(self, docs: list[dict]) -> str:
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs):
            title = doc.get("title", "未知")
            content = doc.get("content", "")
            similarity = doc.get("similarity", 0)
            source = doc.get("source", "")
            weight = doc.get("source_weight", 0.5)
            # 标注来源层级
            tier = "⭐" if weight >= 0.9 else ("◎" if weight >= 0.7 else "○")
            parts.append(
                f"[来源{i + 1}] {tier} {title} "
                f"(来源: {source}, 相关度: {similarity:.0%})\n{content}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, context_docs: list[dict]) -> dict:
        context = self.build_context(context_docs)
        if not context:
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "has_knowledge": False,
            }
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_USER_PROMPT.format(
                context=context, question=question,
            )},
        ]
        answer = self._call_llm(messages)
        sources = []
        for doc in context_docs:
            sources.append({
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "similarity": doc.get("similarity", 0),
                "url": doc.get("url", ""),
                "source_weight": doc.get("source_weight", 0.5),
            })
        return {
            "answer": answer,
            "sources": sources,
            "has_knowledge": True,
        }

    def _call_llm(self, messages: list[dict]) -> str:
        api_key = self.llm_config.get("api_key", "")
        if not api_key:
            return "LLM 服务未配置，请设置 API Key。\n\n您可以在项目根目录创建 .env 文件，设置 DEEPSEEK_API_KEY=your-key。"
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.llm_config["model"],
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1500,
                "stream": False,
            }
            resp = httpx.post(
                f"{self.llm_config['base_url']}/chat/completions",
                json=body, headers=headers, timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return f"抱歉，AI 服务暂时不可用，请稍后重试。（错误：{str(e)[:100]}）"

    def query(self, question: str) -> dict:
        """完整 RAG 查询流程：检索 → 生成"""
        t0 = time.time()
        docs = self.retrieve(question)
        result = self.generate(question, docs)
        result["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        return result


# 单例
rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine
