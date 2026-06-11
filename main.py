import os,uvicorn
from src.config import Config
from src.qa_chain import qa_chain
from src.document_loader import documentProcessor,save_chunks_to_files

def run_api():
    save_chunks_to_files(documentProcessor, './data/documents', './data/chunk_documents')
    print(f"启动 API 服务: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"API 文档: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    # reload=True 会在代码修改后自动重启服务
    uvicorn.run("src.api:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)


def run_terminal():
    while True:
        try:
            question = input("\n请输入您的问题: ")

            if question.lower() in ["quit", "exit"]:
                print("再见！")
                break
            if question.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print("=" * 60)
                print("    企业知识问答系统 - 控制台模式")
                print("=" * 60)
                continue
            if not question.strip():
                continue
            print("\n正在思考...")
            result = qa_chain.answer(question)
            print("【回答】")
            print(result["answer"])
            if result.get("sources"):  # 使用 .get() 防止 sources 为 None 时报错
                print("\n【参考来源】")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n来源 {i}:")
                    # 【关键修复】将键名对齐为你之前 answer 方法返回的字段
                    print(f"文件: {source.get('source_file', '未知来源')}")
                    snippet = source.get('snippet', '')
                    if len(snippet) > 200:
                        print(f"内容片段: {snippet[:200]}...")
                    else:
                        print(f"内容片段: {snippet}")
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n出错了: {str(e)}")


if __name__ == '__main__':
    run_api()
