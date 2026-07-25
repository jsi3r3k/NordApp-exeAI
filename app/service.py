from app.llm import GeminiAnswerGenerator
from app.retrieval import Retriever
from app.schemas import AnswerResponse, SourceResponse

REFUSAL_MESSAGE = (
    "Nie znalazłem tej informacji w bazie wiedzy NordApp."
)

class RAGService: 
    def __init__(
        self,
        retriever: Retriever,
        generator: GeminiAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> AnswerResponse:
        search_results = self.retriever.search(
            question=question, 
            top_k=2,
        )

        generation = self.generator.generate(
            question=question,
            chunks=[
                result.chunk
                for result in search_results
            ],
        )

        results_by_id = {
            result.chunk.chunk_id: result
            for result in search_results
        }

        selected_source_ids = list(
            dict.fromkeys(generation.source_ids)
        )

        sources_are_valid = (
            bool(selected_source_ids)
            and all(
                source_id in results_by_id
                for source_id in selected_source_ids
            )
        )

        if (
            not generation.answerable
            or not generation.answer.strip()
            or not sources_are_valid
        ):
            return AnswerResponse(
                answer=REFUSAL_MESSAGE,
                grounded=False,
                sources=[]
            )

        sources = [
            SourceResponse(
                chunk_id=results_by_id[source_id].chunk.chunk_id,
                section=results_by_id[source_id].chunk.section,
                content=results_by_id[source_id].chunk.content,
                score=round(
                    results_by_id[source_id].score,
                    4
                )
            )
            for source_id in selected_source_ids
        ]

        return AnswerResponse(
            answer=generation.answer.strip(),
            grounded=True,
            sources=sources
        )