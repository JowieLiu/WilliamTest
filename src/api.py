from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AAPL 10-K QA System")

vector_store = None
retriever = None


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
