from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AAPL 10-K QA System")

vector_store = None
retriever = None

# 存储用户反馈
feedback_storage = []


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class ApiConfigRequest(BaseModel):
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = "gpt-3.5-turbo"


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool
    comment: Optional[str] = None


@app.post("/api/qa", response_model=QueryResponse)
async def qa_endpoint(request: QueryRequest):
    global vector_store, retriever
    
    if not vector_store or not retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        from src.qa_generator import get_global_generator
        
        results = retriever.retrieve(request.question, request.top_k)
        context = retriever.format_context(results)
        generator = get_global_generator()
        answer = generator.generate_answer(request.question, context)
        
        return QueryResponse(
            answer=answer,
            sources=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa/stream")
async def qa_stream_endpoint(request: QueryRequest):
    """流式输出接口"""
    global vector_store, retriever
    
    if not vector_store or not retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        from src.qa_generator import get_global_generator
        
        results = retriever.retrieve(request.question, request.top_k)
        context = retriever.format_context(results)
        generator = get_global_generator()
        
        # 先发送源信息
        sources_data = json.dumps({"sources": results}, ensure_ascii=False)
        yield f"data: {sources_data}\n\n"
        
        # 生成答案
        answer = generator.generate_answer(request.question, context)
        
        # 模拟流式输出（逐词发送）
        words = answer.split()
        current_answer = ""
        for word in words:
            current_answer += word + " "
            time.sleep(0.05)  # 模拟网络延迟
            answer_data = json.dumps({"answer": current_answer.strip()}, ensure_ascii=False)
            yield f"data: {answer_data}\n\n"
        
        # 发送完成信号
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"


@app.post("/api/configure-llm")
async def configure_llm(config: ApiConfigRequest):
    global vector_store, retriever
    
    from src.qa_generator import OpenAIApiGenerator, set_global_generator, TinyLlamaGenerator
    
    if config.api_url and config.api_key:
        generator = OpenAIApiGenerator(
            api_url=config.api_url,
            api_key=config.api_key,
            model_name=config.model_name or "gpt-3.5-turbo"
        )
        set_global_generator(generator)
        return {"status": "success", "message": "Configured to use API"}
    else:
        generator = TinyLlamaGenerator()
        set_global_generator(generator)
        return {"status": "success", "message": "Configured to use local TinyLlama"}


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """用户反馈接口"""
    feedback_data = {
        "timestamp": time.time(),
        "question": feedback.question,
        "answer": feedback.answer,
        "helpful": feedback.helpful,
        "comment": feedback.comment
    }
    feedback_storage.append(feedback_data)
    
    # 限制存储数量，只保留最近 1000 条
    if len(feedback_storage) > 1000:
        feedback_storage.pop(0)
    
    return {"status": "success", "message": "Feedback received"}


@app.get("/api/feedback")
async def get_feedback(limit: int = 10):
    """获取用户反馈（管理接口）"""
    return {
        "total": len(feedback_storage),
        "feedback": feedback_storage[-limit:]
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/generator-info")
async def get_generator_info_endpoint():
    from src.qa_generator import get_generator_info
    return get_generator_info()


def initialize_components(vs, ret, qa_gen):
    global vector_store, retriever
    vector_store = vs
    retriever = ret
    from src.qa_generator import set_global_generator
    set_global_generator(qa_gen)
