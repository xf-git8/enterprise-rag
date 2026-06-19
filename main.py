import os,uvicorn,sys
from src.config import Config
from src.qa_chain import qa_chain
from src.vector_store import vector_store_manager
from src.doucments.document_loader import documentProcessor
from src.doucments.document_loader import DocumentProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#在 main.py 最开头添加环境变量（避免 attention 实现错误）
os.environ["TORCH_ATTENTION_IMPLEMENTATION"] = "eager"
# 初始化向量库
def init_vector():
    import argparse
    parser = argparse.ArgumentParser(description="初始化文档向量库")
    parser.add_argument("input_path", nargs='?', default="./data/documents", help="文档路径（默认为 ./data/documents）")
    parser.add_argument("--clear", action="store_true", help="先清空现有向量库")
    args = parser.parse_args()

    if args.clear:
        print("清空现有向量库...")
        vector_store_manager.clear_store()

    print(f"处理文档: {args.input_path}")
    processor = DocumentProcessor()

    try:
        docs = processor.process_documents(args.input_path)
        print(f"文档切分完成，共 {len(docs)} 个片段")
        print("添加到向量数据库...")
        vector_store_manager.add_documents(docs)

        print("文档初始化完成！")
    except Exception as e:
        print(f"处理失败: {e}")
def run_api():
    docs = documentProcessor.process_documents('./data/documents')
    if docs:
        documentProcessor.save_chunks_to_local( docs, './data/chunk_documents')
    else:
        print("警告：没有提取到任何文档内容！")
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
    init_vector()
    run_api()
