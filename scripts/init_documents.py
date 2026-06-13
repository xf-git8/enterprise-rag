# 初始化向量库
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_loader import DocumentProcessor
from src.vector_store import vector_store_manager

def main():
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
        sys.exit(1)


if __name__ == "__main__":
    main()