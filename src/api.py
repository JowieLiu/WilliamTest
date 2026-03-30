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


@app.post("/api/qa", response_model=QueryResponse)
async def qa_endpoint(request: QueryRequest):
    global vector_store, retriever
    
    if not vector_store or not retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        results = retriever.retrieve(request.question, request.top_k)
        
        answer_parts = []
        for i, source in enumerate(results, 1):
            metadata = source['metadata']
            answer_parts.append(f"【{i}. {metadata.get('year', 'N/A')}年 - {metadata.get('section_title', 'N/A')}】")
            answer_parts.append(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
            answer_parts.append("")
        
        answer = "\n".join(answer_parts)
        
        return QueryResponse(
            answer=answer,
            sources=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


def initialize_components(vs, ret, qa_gen=None):
    global vector_store, retriever
    vector_store = vs
    retriever = ret
