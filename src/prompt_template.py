# 定义提示词模板类
from typing import List
from langchain_core.documents import Document

class PromptBuilder:
    def __init__(self):
        self.system_prompt = """
        你是一个专业的企业知识问答助手。请根据提供的参考文档，回答用户问题：
        规则：
            1.仅使用提供的参考文档内容进行回答
            2.如果文档中没有相关信息，请明确说“根据现有文档，无法回答该问题”
            3.回答问题要简介准确，不要添加猜测内容
            4.如果问题涉及多个方面，请分点回答
            5.回答语言要正式、专业        
        """

    def build_prompt(self, question: str, context_docs: List[Document]) -> str:
        """普通方法 构建完整的提示词方法"""
        # 调用本类私有方法返回文档相关内容信息
        context = self._format_context(context_docs)
        # 拼接提示词
        prompt = f"""
{self.system_prompt}
参考文档:
{context}
用户问题:
{question}
请根据以上文档回答用的问题:

"""     # 去除前后空格并返回拼接好的提示词内容
        return prompt.strip()

    def _format_context(self, docs: list[Document]):
        """私有方法 格式化上下文文档"""
        # 定义列表
        formatted = []
        for i, doc in enumerate(docs):
            # 从文档的元数据提取来源信息，如果不存在默认为unknow
            source = doc.metadata.get("source", "unknow")
            # 提取文档片段ID,如果不存在则使用当前索引i作为默认值
            chunk_id = doc.metadata.get("chunk_id", i)
            # 获取文本实际内容
            context = doc.page_content
            # 将当前文档的信息格式化为多行字符串并追加到列表中
            formatted.append(f"文档{i + 1}(来源：{source}，片段：{chunk_id}:{context})")
            # 将所有格式化后的文档字符串用换行符拼接，并去除首尾多余空白后返回
            return "\n".join(formatted).strip()
