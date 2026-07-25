from unittest.mock import Mock

from app.chunking import Chunk
from app.retrieval import SearchResult
from app.schemas import GroundedGeneration
from app.service import REFUSAL_MESSAGE, RAGService


def create_service(
    generation: GroundedGeneration,
) -> tuple[RAGService, Mock, Mock]:
    chunk = Chunk(
        chunk_id="plany-i-ceny",
        section="Plany i ceny",
        content="Plan Team kosztuje 149 zł miesięcznie.",
    )

    search_result = SearchResult(
        chunk=chunk,
        score=0.8675,
    )

    retriever = Mock()
    retriever.search.return_value = [search_result]

    generator = Mock()
    generator.generate.return_value = generation

    service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    return service, retriever, generator


def test_returns_grounded_answer_with_valid_source() -> None:
    generation = GroundedGeneration(
        answerable=True,
        answer="Plan Team kosztuje 149 zł miesięcznie.",
        source_ids=["plany-i-ceny"],
    )
    service, retriever, generator = create_service(generation)

    response = service.answer("Ile kosztuje plan Team?")

    assert response.grounded is True
    assert response.answer == generation.answer
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "plany-i-ceny"
    assert response.sources[0].score == 0.8675

    retriever.search.assert_called_once_with(
        question="Ile kosztuje plan Team?",
        top_k=2,
    )
    generator.generate.assert_called_once()


def test_returns_refusal_when_context_has_no_answer() -> None:
    generation = GroundedGeneration(
        answerable=False,
        answer="",
        source_ids=[],
    )
    service, _, _ = create_service(generation)

    response = service.answer(
        "Czy NordApp integruje się z Jira?"
    )

    assert response.answer == REFUSAL_MESSAGE
    assert response.grounded is False
    assert response.sources == []


def test_rejects_answer_with_unknown_source_id() -> None:
    generation = GroundedGeneration(
        answerable=True,
        answer="NordApp integruje się z Jira.",
        source_ids=["nieistniejace-zrodlo"],
    )
    service, _, _ = create_service(generation)

    response = service.answer(
        "Czy NordApp integruje się z Jira?"
    )

    assert response.answer == REFUSAL_MESSAGE
    assert response.grounded is False
    assert response.sources == []