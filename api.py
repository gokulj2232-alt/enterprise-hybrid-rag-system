from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time

from hybrid_search import HybridRAGSearchEngine


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI-Powered Enterprise Hybrid RAG API",
    description="Backend API for Enterprise Hybrid Search and RAG",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="User question"
    )

    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieval results"
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class AskResponse(BaseModel):

    success: bool

    query: str

    answer: str

    processing_time_seconds: float

    rag_details: dict


# ============================================================
# GLOBAL RAG ENGINE
# ============================================================

rag_engine = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global rag_engine

    print()
    print("=" * 70)
    print("STARTING ENTERPRISE RAG API")
    print("=" * 70)

    print()
    print("Loading Hybrid RAG Search Engine...")
    print()

    try:

        rag_engine = HybridRAGSearchEngine()

        print()
        print("=" * 70)
        print("HYBRID RAG SEARCH ENGINE LOADED")
        print("=" * 70)
        print()

    except Exception as error:

        print()
        print("=" * 70)
        print("RAG ENGINE STARTUP ERROR")
        print("=" * 70)
        print(error)
        print("=" * 70)

        raise


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Enterprise Hybrid RAG API",
        "rag_engine_loaded": rag_engine is not None
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI-Powered Enterprise Hybrid RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ask": "/ask"
        }
    }


# ============================================================
# ASK API
# ============================================================

@app.post("/ask")
def ask_question(request: AskRequest):

    global rag_engine

    if rag_engine is None:

        raise HTTPException(
            status_code=503,
            detail="RAG engine is not loaded."
        )

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    start_time = time.time()

    print()
    print("=" * 70)
    print("API REQUEST")
    print("=" * 70)

    print("Query:", query)
    print("Top-K:", request.top_k)

    try:

        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        result = rag_engine.ask(query)

        processing_time = round(
            time.time() - start_time,
            4
        )

        # ----------------------------------------------------
        # EXTRACT ANSWER
        # ----------------------------------------------------

        if isinstance(result, dict):

            answer = (
                result.get("answer")
                or result.get("final_answer")
                or result.get("response")
                or "I don't know based on the provided documents."
            )

        elif isinstance(result, str):

            answer = result

        else:

            answer = str(result)

        # ----------------------------------------------------
        # BASIC RAG DETAILS
        # ----------------------------------------------------

        rag_details = {

            "system": "Enterprise Hybrid RAG",

            "retrieval": {
                "semantic_search": True,
                "bm25_search": True,
                "hybrid_retrieval": True,
                "cross_encoder_reranking": True
            },

            "generation": {
                "rag_generation": True,
                "answerability_gate": True,
                "grounding_validation": True
            },

            "query": query,

            "top_k": request.top_k,

            "processing_time_seconds": processing_time,

            "raw_result": result
        }

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "query": query,

            "answer": answer,

            "processing_time_seconds": processing_time,

            "rag_details": rag_details
        }

        print()
        print("Answer:", answer)

        print(
            "Processing Time:",
            processing_time,
            "seconds"
        )

        print("=" * 70)
        print()

        return response

    except Exception as error:

        print()
        print("=" * 70)
        print("API ERROR")
        print("=" * 70)

        print(error)

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(error)}"
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )