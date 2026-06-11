# 定义API功能接口
import shutil,os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from src import qa_chain, Config, DocumentProcessor, vector_store_manager

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


app = FastAPI(title="企业知识问答系统", description="基于RAG的企业知识问答API")


@app.post("/api/qa", tags=["问答接口"])
async def ask_question(request: QuestionRequest):
    try:
        result = qa_chain.answer(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload", tags=["上传文档"])
async def upload_documents(files: List[UploadFile] = File(...)):
    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    saved_files = []
    document_types = ['.txt', '.pdf', '.doc', '.docx']
    # 遍历判断文档类型
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in document_types:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
        file_path = os.path.join(Config.DOCUMENTS_DIR, file.filename)
        # 写入文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
    # 文档进行切割
    processor = DocumentProcessor()
    docs = processor.process_documents(Config.DOCUMENTS_DIR)
    vector_store_manager.add_documents(docs)
    return {
        "message": "文档上传成功",
        "files": saved_files,
        "documents_count": len(docs),
    }


@app.delete("/api/documents/clear", tags=["清空文档"])
async def clear_documents():
    vector_store_manager.clear_store()
    return {"message": "向量数据库已清空"}


@app.get("/api/statistics", tags=["统计信息"])
async def get_statistics():
    count = vector_store_manager.count_documents()
    return {"documents_count": count}

@app.get("/api/health", tags=["健康检查"])
async def health_check():
    return {"status": "healthy"}