from pydantic import BaseModel, field_validator


class ResearchRequest(BaseModel):
    company: str

    @field_validator("company")
    @classmethod
    def company_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("company name must not be empty")
        if len(v) > 200:
            raise ValueError("company name must be 200 characters or fewer")
        return v


class SourceItem(BaseModel):
    title: str
    url: str
    dimension: str


class RawResult(BaseModel):
    title: str
    url: str
    content: str
    score: float


class ValidationResult(BaseModel):
    is_valid: bool = True
    ungrounded_claims: list[str] = []
    incomplete_sections: list[str] = []
    no_data_sections: list[str] = []
    overall_score: float = 1.0


class ResearchResponse(BaseModel):
    company: str
    brief: str
    sources: list[SourceItem]
    changes: list[str]
    confidence_scores: dict[str, int | None]
    is_first_run: bool
    timestamp: str
    raw_results: dict[str, list[RawResult]] | None = None
    validation: ValidationResult = ValidationResult()


class CachedReport(BaseModel):
    company: str
    brief: str
    sections: dict[str, str]
    timestamp: str
