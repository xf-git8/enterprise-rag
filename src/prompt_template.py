from typing import List
from langchain_core.documents import Document
class PromptBuilder:
    def __init__(self):
        # 【修改点1】增加关于引用的强制指令
        self.system_prompt = """
        你是一个专业的企业知识问答助手。请根据提供的参考文档，回答用户问题。
        【核心规则】：
        1. 仅使用提供的参考文档内容进行回答。
        2. 如果文档中没有相关信息，请明确说“根据现有文档，无法回答该问题”。
        3. 【重要】在回答的每一句话或关键事实后，必须用方括号标注来源编号，格式为 [1], [2] 等。
           例如：“员工年假为5天[1]。请假需提前3天申请[2]。”
        4. 不要编造任何文档中不存在的信息。
        """

    def build_prompt(self, question: str, context_docs: List[Document]) -> str:
        context = self._format_context(context_docs)
        prompt = f"""
{self.system_prompt}

参考文档列表：
{context}

用户问题：
{question}

请结合上述文档回答问题，并严格遵守引用标注规则：
"""
        return prompt.strip()

    def _format_context(self, docs: list[Document]):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            # 【修改点2】简化格式，直接使用 [i+1] 作为开头，方便模型识别和正则提取
            # 这里的 i+1 就是模型需要输出的引用编号
            formatted.append(f"[{i + 1}] 来源：{source}\n内容：{doc.page_content}")

        return "\n\n".join(formatted).strip()

prompt_build = PromptBuilder()