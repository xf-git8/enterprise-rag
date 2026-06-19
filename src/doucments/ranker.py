# src/ranker.py
import numpy as np
import jieba
from typing import List, Dict, Any

class Ranker:
    def __init__(self):
        self.idf_dict = {}
        self.avg_doc_len = 0
        self.doc_count = 0

    # ---- 中文分词 ----
    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    # ---- 预计算语料库级别的 IDF ----
    def build_idf(self, all_docs: List[str]):
        """在检索完成后、重排序之前调用一次"""
        self.doc_count = len(all_docs)
        self.doc_freq = {}
        total_len = 0

        for doc in all_docs:
            tokens = set(self._tokenize(doc))  # set 去重，每个词只计一次
            total_len += len(self._tokenize(doc))
            for token in tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = total_len / max(self.doc_count, 1)

        for token, freq in self.doc_freq.items():
            self.idf_dict[token] = np.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1
            )

    # ---- 修复后的 BM25 ----
    def _bm25_score(self, query: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(doc)
        doc_len = len(doc_tokens)

        score = 0.0
        for token in query_tokens:
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            idf = self.idf_dict.get(token, 0)
            numerator = idf * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
            score += numerator / denominator
        return score

    # ---- RRF 混合排序（推荐替代原来的 hybrid_rerank）----
    def rrf_rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion：无需归一化，直接按排名融合
        比加权求和更稳定，业界主流方案
        """
        n = len(docs)
        rrf_scores = {i: 0.0 for i in range(n)}

        # 按向量分数排序，贡献 RRF 分
        vector_ranked = sorted(range(n), key=lambda i: docs[i].get('vector_score', 0), reverse=True)
        for rank, idx in enumerate(vector_ranked):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        # 按 BM25 分数排序，贡献 RRF 分
        bm25_ranked = sorted(range(n), key=lambda i: self._bm25_score(query, docs[i].get('content', '')), reverse=True)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        # 按 RRF 总分排序
        for i, doc in enumerate(docs):
            doc['rrf_score'] = rrf_scores[i]

        docs.sort(key=lambda x: x['rrf_score'], reverse=True)
        return docs[:top_k]
ranker = Ranker()