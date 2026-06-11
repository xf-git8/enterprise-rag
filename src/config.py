# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    # 支持ocr pdf扫描件的poppler's install path
    POPPLER_BIN_PATH = os.getenv("POPPLER_BIN_PATH")
    # 支持ocr pdf扫描件的tesseract's install path
    TESSERACT_BIN_PATH = os.getenv("TESSERACT_BIN_PATH")

    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", 8080))
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")
    TOP_K = int(os.getenv("TOP_K", 5))

    @classmethod
    def validate(cls):
        """根据当前选择的 LLM 提供商，校验对应的环境变量是否已配置"""
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("❌ 缺少 OPENAI_API_KEY，请在 .env 中配置")
        if cls.LLM_PROVIDER == "dashscope" and not cls.DASHSCOPE_API_KEY:
            raise ValueError("❌ 缺少 DASHSCOPE_API_KEY，请在 .env 中配置")


config = Config()