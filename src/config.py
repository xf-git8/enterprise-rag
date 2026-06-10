# 全局模型配置类
import os
# 导入环境变量加载库
from dotenv import load_dotenv
# 执行加载操作，
# 使 .env 文件中的变量可以在当前 Python 环境中通过 os.getenv() 获取
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    API_HOST = os.getenv("API_HOST")
    API_PORT = int(os.getenv("API_PORT"))
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")
    TOP_K = int(os.getenv("TOP_K"))

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is openai")
        if cls.LLM_PROVIDER == "dashscope" and not cls.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is required when LLM_PROVIDER is dashscope")
config = Config()