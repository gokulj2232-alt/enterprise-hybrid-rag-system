
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import rag_pipeline


app = FastAPI(
    title="Enterprise Hybrid RAG API",
    description="AI-powered document question answering API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise Hybrid RAG API",
        "rag_engine_loaded": True
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        return {
            "question": "",
            "answer": "Please provide a question.",
            "sources": [],
            "processing_time_seconds": 0
        }

    start_time = time.time()

    result = rag_pipeline(
        question,
        top_k=request.top_k
    )

    processing_time = round(time.time() - start_time, 3)

    return {
        "question": question,
        "answer": result.get(
            "answer",
            "I don't know based on the provided documents."
        ),
        "sources": result.get("sources", []),
        "processing_time_seconds": processing_time
    }
