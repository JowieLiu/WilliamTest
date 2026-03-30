from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AAPL 10-K QA System")

vector_store = None
retriever = None
qa_generator = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.post("/api/qa", response_model=QueryResponse)
async def qa_endpoint(request: QueryRequest):
    global vector_store, retriever, qa_generator
    
    if not vector_store or not retriever or not qa_generator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        results = retriever.retrieve(request.question, request.top_k)
        context = retriever.format_context(results)
        answer = qa_generator.generate_answer(request.question, context)
        
        return QueryResponse(
            answer=answer,
            sources=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


def initialize_components(vs, ret, qa_gen):
    global vector_store, retriever, qa_generator
    vector_store = vs
    retriever = ret
    qa_generator = qa_gen
