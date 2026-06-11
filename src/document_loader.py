import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30) -> None:
        """初始化文档处理器，配置核心参数和切割规则"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # 【新增】用于保证跨文件、跨批次切分时 chunk_id 的全局唯一性
        self._global_chunk_counter = 0

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件路径，读取并切割单个文档"""
        ext_name = os.path.splitext(file_path)[1].lower()

        if ext_name == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext_name == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext_name == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext_name}")

        documents = loader.load()
        return self.split_documents(documents)

    def load_directory(self, directory_path: str) -> List[Document]:
        """根据目录路径，递归读取并切割目录下所有支持的文档"""
        try:
            loader = DirectoryLoader(directory_path, glob="**/*", show_progress=True)
            documents = loader.load()
            return self.split_documents(documents)
        except Exception as e:
            print(f"Error loading directory: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将多个文档内容切割成带有元数据的小文本块
        【优化】方法名修正为复数形式，并增强 chunk_id 的唯一性
        """
        if self.text_splitter is None:
            raise ValueError("Text splitter is not initialized.")

        split_docs = self.text_splitter.split_documents(documents)

        # 【优化】使用全局计数器，确保即使批量处理几百个文件，ID也不会冲突
        for doc in split_docs:
            doc.metadata["chunk_id"] = self._global_chunk_counter
            self._global_chunk_counter += 1

        return split_docs

    def process_documents(self, input_path: str) -> List[Document]:
        """统一的文档处理入口方法：智能判断输入路径是文件还是文件夹"""
        if os.path.isfile(input_path):
            docs = self.load_document(input_path)
        elif os.path.isdir(input_path):
            docs = self.load_directory(input_path)
        else:
            raise ValueError(f"Invalid path: {input_path}")

        return docs