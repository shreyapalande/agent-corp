from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_valid: bool
    ungrounded_claims: list[str] = field(default_factory=list)
    incomplete_sections: list[str] = field(default_factory=list)
    no_data_sections: list[str] = field(default_factory=list)
    overall_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "ungrounded_claims": self.ungrounded_claims,
            "incomplete_sections": self.incomplete_sections,
            "no_data_sections": self.no_data_sections,
            "overall_score": self.overall_score,
        }

    @staticmethod
    def from_dict(d: dict) -> "ValidationResult":
        return ValidationResult(
            is_valid=d.get("is_valid", True),
            ungrounded_claims=d.get("ungrounded_claims", []),
            incomplete_sections=d.get("incomplete_sections", []),
            no_data_sections=d.get("no_data_sections", []),
            overall_score=d.get("overall_score", 1.0),
        )
