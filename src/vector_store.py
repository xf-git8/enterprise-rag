import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

from src.config import Config, config


# 存入向量数据库类
class VectorStoreManager:
    def __init__(self):
        """ 初始化函数：构建嵌入模型实例并初始化向量存储。"""
        # 1. 根据全局配置加载 SentenceTransformer 嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = None
        self._init_store()

    # 定义私有方法在本类中调用
    def _init_store(self):
        """配置向量数据库和embedding模型"""
        os.makedirs('VECTOR_DB_PATH', exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """延迟初始化加载文档     向向量数据库中批量添加文档。"""
        # 如果 self.vector_store 为 None 或 False，则触发懒加载机制
        if not self.vector_store:
            # 调用内部初始化方法，确保 Chroma 实例和 Embedding 模型已就绪
            self._init_store()
        return self.vector_store.add_documents(documents)

    def delete_documents(self, ids: List[str]) -> None:
        """根据 ids 批量删除文档列表"""
        if not self.vector_store:
            self._init_store()
        self.vector_store.delete(ids)

    def search_documents(self, query: str, top_k: int) -> List[Document]:
        """
        跟据query获取相关的top-k片段
        :param query:
        :param top_k:
        :return: 文档列表List[chunk1,chunk2....]
        """
        if not self.vector_store:
            self._init_store()
        if top_k is None:
            top_k = config.TOP_K

        return self.vector_store.similarity_search(query=query, k=top_k)

    def search_documents_score(self, query: str, top_k: int) -> List[tuple]:
        """
        跟据query获取相关的top-k片段和相关度分数
        :param query:
        :param top_k:
        :return: List[tuple] 元组的文档和分数列表[(chunk1,score1),(chunk2,score2).....]
        """
        if not self.vector_store:
            self._init_store()
        if top_k is None:
            top_k = config.TOP_K
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

    def clear_store(self) -> None:
        """
         清空并重置整个向量数据库。
        :return:
        """
        if self.vector_store:
            self.vector_store.delete_collection()
            self._init_store()

    def count_store(self) -> int:
        """
        统计改向量数据库文档的总数
        ChromaDB 在较新版本中已将 .count() 重命名为 .count_documents()
        :return: int 总数
        """
        if not self.vector_store:
            self._init_store()
        collections = self.vector_store.count_documents()
        # 建议加上 hasattr 判断以兼容不同版本的 ChromaDB
        if hasattr(collections, 'count_documents'):
            return collections.count_documents()
        else:
            return collections.count()


# 创建 向量数据库对象供使用
vector_store_manger = VectorStoreManager()
