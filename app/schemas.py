from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="Question about the NordApp knowledge base.",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Question cannot be blank.")

        return cleaned_value


class SourceResponse(BaseModel):
    chunk_id: str
    section: str
    content: str
    score: float = Field(ge=-1.0, le=1.0)


class AnswerResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceResponse]


class GroundedGeneration(BaseModel):
    answerable: bool = Field(
        description=(
            "True only when the supplied context explicitly contains "
            "enough information to answer the question."
        )
    )
    answer: str = Field(
        description=(
            "Answer based exclusively on the supplied context, "
            "or an empty string when the question is not answerable."
        )
    )
    source_ids: list[str] = Field(
        description=(
            "IDs of context chunks that directly support the answer. "
            "Empty when the question is not answerable."
        )
    )