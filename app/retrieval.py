from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.chunking import Chunk

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one chunk.")
        
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        passages = [
            f"passage: {chunk.section}\n{chunk.content}"
            for chunk in chunks
        ]

        self.chunk_embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def search(
        self,
        question: str,
        top_k: int = 2
    ) -> list[SearchResult]:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [f"query: {cleaned_question}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.chunk_embeddings @ query_embedding
        result_count = min(top_k, len(self.chunks))
        best_indicies = np.argsort(scores)[::-1][:result_count]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in best_indicies
        ] 