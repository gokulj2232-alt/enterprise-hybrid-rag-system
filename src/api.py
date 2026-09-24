import sys
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# RAG PIPELINE
# ============================================================

from src.rag.rag_pipeline import rag_pipeline


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Hybrid RAG API",
    description="AI-powered Enterprise Hybrid RAG question answering API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="User question"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of source documents to return"
    )

    conversation: list[dict] = Field(
        default=[],
        description="Previous conversation messages"
    )

    uploaded_dataset: bool = Field(
        default=False,
        description="Use the currently uploaded dataset"
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Enterprise Hybrid RAG API",
        "version": "1.0.0",
        "status": "running"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Enterprise Hybrid RAG API",
        "rag_engine_loaded": True
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        return {
            "question": "",
            "answer": "Please provide a question.",
            "sources": [],
            "processing_time_seconds": 0,
            "conversation_messages_received": 0
        }

    start_time = time.time()

    # --------------------------------------------------------
    # RUN RAG PIPELINE
    # --------------------------------------------------------

    result = rag_pipeline(
        question,
        top_k=request.top_k,
        conversation=request.conversation,
        uploaded_dataset=request.uploaded_dataset
    )

    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    processing_time = round(
        time.time() - start_time,
        3
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "question": question,

        "answer": result.get(
            "answer",
            "I don't know based on the provided documents."
        ),

        "sources": result.get(
            "sources",
            []
        ),

        "processing_time_seconds": processing_time,

        "conversation_messages_received": len(
            request.conversation
        ),

        "uploaded_dataset": request.uploaded_dataset
    }