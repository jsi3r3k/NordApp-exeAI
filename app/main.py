import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from app.chunking import load_markdown, split_by_headings
from app.llm import DEFAULT_MODEL_NAME, GeminiAnswerGenerator
from app.retrieval import Retriever
from app.schemas import AnswerResponse, QuestionRequest
from app.service import RAGService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "data" / "nordapp_baza_wiedzy.md"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is required."
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_MODEL_NAME,
    )

    document = load_markdown(KNOWLEDGE_BASE_PATH)
    chunks = split_by_headings(document)

    retriever = Retriever(chunks)
    generator = GeminiAnswerGenerator(
        api_key=api_key,
        model_name=model_name,
    )

    app.state.rag_service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    try:
        yield
    finally:
        generator.close()


app = FastAPI(
    title="NordApp RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    payload: QuestionRequest,
    request: Request,
) -> AnswerResponse:
    service: RAGService = request.app.state.rag_service
    return service.answer(payload.question)