# llm_client.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .config import Config


class LLMClient:
    """ 大语言模型（LLM）客户端封装类 """

    def __init__(self):
        # 在初始化前校验环境变量配置是否完整
        Config.validate()
        # 根据配置创建具体的模型客户端实例
        self.client = self._create_client()

    def _create_client(self):
        if Config.LLM_PROVIDER == "dashscope":
            return ChatOpenAI(
                api_key=Config.DASHSCOPE_API_KEY,

                # 【关键修复 1】：必须显式指定阿里云的 Base URL，防止请求被错误路由到 OpenAI
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

                # 【关键修复 2】：修正模型名称，qwen3.7-plus 不存在，改为 qwen-plus
                model_name="qwen-plus",

                temperature=0.2,
                max_tokens=2048
            )

        # 如果后续要接入 OpenAI，可以在这里加 elif Config.LLM_PROVIDER == "openai": ...
        raise ValueError(f"❌ 不支持的 LLM 提供商: {Config.LLM_PROVIDER}")

    def generate(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]

        try:
            response = self.client.invoke(messages)
            # 正常情况：LangChain 返回的是 AIMessage 对象
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()

        except Exception as e:
            # 打印日志后重新抛出，方便调试
            print(f"❌ LLM 调用失败: {type(e).__name__} - {e}")
            raise e

        # 实例化单例对象


llm_client = LLMClient()