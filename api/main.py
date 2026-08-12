"""
Optional REST API layer (FastAPI) exposing the same capabilities as the
Streamlit UI, for programmatic / integration use.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agents.agent_orchestrator import AgentOrchestrator
from core.exceptions import AppBaseException
from services.document_service import process_document
from services.vector_store_service import add_documents, collection_document_count
from utils.logger import logger

app = FastAPI(
    title="Enterprise Document Q&A API",
    description="Upload enterprise documents and ask natural-language questions "
                "over them using a Retrieval-Augmented Generation agent pipeline.",
    version="1.0.0",
)

orchestrator = AgentOrchestrator()


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": collection_document_count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        chunks = process_document(file.filename, file_bytes)
        added = add_documents(chunks)
        return {"filename": file.filename, "chunks_added": added}
    except AppBaseException as exc:
        logger.warning(f"Upload failed: {exc.message}")
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected upload error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest):
    try:
        result = orchestrator.run(payload.question)
        return QuestionResponse(answer=result.answer, sources=result.sources, grounded=result.grounded)
    except AppBaseException as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected ask error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
