# 定义文档加载器分割类

import os
from typing import List

# 导入 LangChain 提供的多种文档加载器，用于处理不同格式的本地文件
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
# 导入 LangChain 提供的文档对象，用于封装文档内容
from langchain_core.documents import Document
# 递归文本切割对象
from langchain_text_splitters import RecursiveCharacterTextSplitter


#  文档处理器类：负责将本地的各类文档（如 PDF, Word, TXT）读取、清洗，
#  并按照指定的规则切分成适合大模型和向量数据库处理的小文本块（Chunks）
class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200) -> None:
        """
        初始化文档处理器，配置核心参数和切割规则
        :param chunk_size:切割字符大小
        :param chunk_overlap:相邻文本块之间的重叠字符数
        """
        self.text_spliterself = None
        self.text_splitter = None
        self.text_spliterself.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # 按优先级尝试分隔符：先按双换行（段落）切，再按单换行，最后按空格或单字符硬切
            separators=["\n\n", "\n", " ", ""]
        )

    # 文档加载函数：根据文件路径，读取并切割文档
    def load_document(self, file_path: str) -> List[Document]:
        """
        根据文件路径，读取并切割文档
        :param file_path: 文件路径
        :return: 文档列表[包含文档内容和基础元数据的对象列表]
        """
        # 以点分割文件名称，获取文件扩展名，
        ext_name = os.path.splitext(file_path)[1].lower()
        # 根据文件扩展名，选择合适的文档加载器
        if ext_name == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext_name == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext_name == ".txt":
            loader = TextLoader(file_path, "encoding='utf-8'")
        else:
            # 如果文件扩展名不在预设范围内，则抛出异常
            raise ValueError(f"Unsupported file type:{ext_name}")

    # 加载读取文件
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        根据目录路径，读取并且目录下的所有文档
        :param directory_path:
        :return: 目录下所有文档解析后的Document对象列表
        """
        # glob="**/*" 表示递归查找该目录及其子目录下的所有文件 添加异常避免加载空文件
        try:
            loader = DirectoryLoader(directory_path, glob="**/*")
            return loader.load()
        except Exception as e:
            print(f"Error loading directory: {e}")
            return []

    # 文档切割函数：将文档内容切割成多个小文本块
    def split_document(self, document: Document) -> List[Document]:
        """
        将文档内容切割成多个小文本块
        :param document:
        :return: 切分后带有新元数据的文档块列表
        """
        if self.text_splitter is None:
            raise ValueError("Text splitter is not initialized.")
        else:
            split_docs = self.text_splitter.split_documents([document])
            # 遍历切分后的文档，为每一个文本块添加一个递增的 chunk_id 这有助于在后续检索时追踪文本块的顺序和位置
            for i, document in enumerate(split_docs):
                document.metadata["chunk_id"] = i
        return split_docs
    # 处理切割过程，可直接用向量化的文档块列表
    def process_documents(self, input_path: str) -> List[Document]:
        """
        统一的文档处理入口方法：智能判断输入路径是文件还是文件夹，
        完成从“加载”到“切分”的完整流水线操作。

        :param input_path: 用户传入的文件路径或文件夹路径。
        :return: 最终处理完毕、可直接用于向量化（Embedding）的文档块列表。
        """
        if os.path.isfile(input_path):
            docs = self.load_document(input_path)  # 如果是文件，走单文件加载逻辑
        elif os.path.isdir(input_path):
            docs = self.load_directory(input_path)  # 如果是文件夹，走批量加载逻辑
        else:
            raise ValueError(f"Invalid path: {input_path}")  # 路径无效则报错