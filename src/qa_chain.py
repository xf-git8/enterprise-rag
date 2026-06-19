# src/qa_chain.py
import re
import hashlib
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.config import config
from src.llm_client import llm_client
from src.prompt_template import prompt_build
from src.vector_store import vector_store_manager
from src.doucments.ranker import Ranker,ranker

logger = logging.getLogger(__name__)


class QAChain:
    def __init__(self):
        self.llm = llm_client
        self.prompt_build = prompt_build
        self.vector_store_manager = vector_store_manager
        self.ranker = Ranker()

    def retrieve_context(self, question: str, top_k: int = None) -> List[Document]:
        k = top_k or config.TOP_K
        return self.vector_store_manager.search_documents(question, top_k=k)

    def _extract_citations(self, raw_answer: str) -> List[int]:
        """增强版引用解析，兼容多种格式"""
        patterns = [
            r'\[(\d+)\]',    # [1] [2]
            r'【(\d+)】',    # 【1】【2】
        ]
        indices = set()
        for pattern in patterns:
            indices.update(int(m) for m in re.findall(pattern, raw_answer))
        return sorted(indices)

    def _snippet_fingerprint(self, content: str) -> str:
        """用哈希做指纹，比截断字符串更可靠"""
        return hashlib.md5(content[:300].encode()).hexdigest()

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        完整调用链路：粗召回 → 重排序 → Prompt组装 → LLM生成 → 引用解析
        """
        final_k = top_k or config.TOP_K

        # ====== 阶段一：粗召回（多取 3 倍） ======
        recall_k = final_k * 3
        context_docs = self.retrieve_context(question, top_k=recall_k)

        if not context_docs:
            return {"answer": "未找到相关信息，请换个问题试试。", "sources": []}

        # ====== 阶段二：重排序（精排） ======
        # 2.1 预计算 IDF
        all_contents = [doc.page_content for doc in context_docs]
        self.ranker.build_idf(all_contents)

        # 2.2 构造 Ranker 输入
        docs_for_rerank = []
        for i, doc in enumerate(context_docs):
            docs_for_rerank.append({
                "id": i,
                "content": doc.page_content,
                "vector_score": doc.metadata.get("score", 0),
                "doc": doc,  # 保留原始 Document 对象
            })

        # 2.3 执行 RRF 重排序
        reranked = self.ranker.rrf_rerank(question, docs_for_rerank, top_k=final_k)

        # 2.4 取重排后的 Document 列表
        final_docs = [item["doc"] for item in reranked]

        # ====== 阶段三：Prompt 组装 + LLM 生成 ======
        try:
            prompt = self.prompt_build.build_prompt(question, final_docs)
            raw_answer = self.llm.generate(prompt)

            if not raw_answer or not raw_answer.strip():
                return {"answer": "模型未返回有效回答，请重试。", "sources": []}

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"answer": f"回答生成失败：{str(e)}", "sources": []}

        # ====== 阶段四：引用解析 + 去重 ======
        cited_indices = self._extract_citations(raw_answer)
        seen_fingerprints = set()
        sources_list = []

        for idx in cited_indices:
            doc_idx = idx - 1  # 转为 0-based
            if 0 <= doc_idx < len(final_docs):
                doc = final_docs[doc_idx]
                fingerprint = self._snippet_fingerprint(doc.page_content)

                if fingerprint not in seen_fingerprints:
                    seen_fingerprints.add(fingerprint)
                    sources_list.append({
                        "index": idx,
                        "source_file": doc.metadata.get("source", "Unknown"),
                        "snippet": doc.page_content[:150].strip() + "..."
                    })

        return {
            "answer": raw_answer,
            "sources": sources_list
        }


qa_chain = QAChain()