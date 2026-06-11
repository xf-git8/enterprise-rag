# 问答链核心管理类
# 负责串联向量数据库检索、Prompt 组装和大语言模型生成
# 实现端到端的基于外部知识库的问答功能
import re
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.config import config
from src.llm_client import llm_client
from src.prompt_template import prompt_build
from src.vector_store import vector_store_manager


class QAChain:
    def __init__(self):
        """初始化问答链所需的各个基础组件"""
        # 大语言模型客户端，用于文本生成
        self.llm = llm_client
        # Prompt 构建器，用于格式化输入模板
        self.prompt_build = prompt_build
        # 向量存储管理器，用于文档检索
        self.vector_store_manager = vector_store_manager

    def retrieve_context(self, question: Optional[str] = None, top_k: int = None) -> List[Document]:
        """
        从向量数据库中检索与用户问题相关的上下文文档片段。
        :param question: 用户问题
        :param top_k: 需要返回的最相关文档数量。若未传入则使用全局配置。
        :return: List[document]
        """
        k = top_k or config.TOP_K
        return self.vector_store_manager.search_documents(question, top_k=k)

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        调用大模型生成回答
        """
        # 1. 检索与问题相关的上下文文档
        context_docs = self.retrieve_context(question, top_k)

        # 2. 构建 Prompt
        prompt = self.prompt_build.build_prompt(question, context_docs)

        # 3. 调用大模型
        raw_answer = llm_client.generate(prompt)

        # 4. 【核心逻辑】动态解析引用来源并去重
        cited_indices = set(re.findall(r'\[(\d+)\]', raw_answer))

        seen_snippets = set()  # 用于记录已经出现过的摘要内容（指纹）
        sources_list = []

        for idx_str in sorted(cited_indices, key=int):
            idx = int(idx_str) - 1  # 转为0-based索引

            # 边界检查
            if 0 <= idx < len(context_docs):
                doc = context_docs[idx]
                # 生成摘要指纹
                snippet = doc.page_content[:150].strip() + "..."

                # 【关键修改】只有当这个片段之前没出现过时，才处理
                if snippet not in seen_snippets:
                    # 1. 标记为已见
                    seen_snippets.add(snippet)

                    # 2. 添加到最终列表 (注意缩进，必须在 if 内部)
                    sources_list.append({
                        "index": int(idx_str),
                        "source_file": doc.metadata.get("source", "Unknown"),
                        "snippet": snippet
                    })

        # 5. 返回结果
        return {
            "answer": raw_answer,
            "sources": sources_list
        }

qa_chain = QAChain()
