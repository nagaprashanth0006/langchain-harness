from __future__ import annotations
import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class QueryCategory(str, Enum):
    EMPIRICAL_FACTUAL = "empirical_factual"
    PROCEDURAL_REFERENCE = "procedural_reference"
    INTERPRETIVE_SUBJECTIVE = "interpretive_subjective"
    PERSONAL_SENSITIVE = "personal_sensitive"


class SourceType(str, Enum):
    PEER_REVIEWED = "peer-reviewed"
    PREPRINT = "preprint"
    ARTICLE = "article"
    WEB = "web"


class ClaimStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    FAILED = "failed"
    CONTESTED = "contested"


class OperationMode(str, Enum):
    NORMAL = "normal"
    RESEARCH = "research"


class ResearchDepth(int, Enum):
    SHALLOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    DEEP = 5


class Source(BaseModel):
    id: str = ""
    origin: List[str] = Field(default_factory=list)
    title: str = ""
    text: str = ""
    url: str = ""
    type: SourceType = SourceType.WEB
    retrieved_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def compute_id(self) -> "Source":
        if not self.id:
            normalized = self.text.lower().strip()
            self.id = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin,
            "title": self.title,
            "url": self.url,
            "type": self.type.value,
            "text_preview": self.text[:300] + ("..." if len(self.text) > 300 else ""),
            "retrieved_at": self.retrieved_at,
        }


class Claim(BaseModel):
    id: str
    text: str
    cited_source_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.PENDING
    confidence: float = 0.0


class JudgeVerdict(BaseModel):
    judge_id: str
    model: str
    target: str  # claim_id or "answer"
    score: float = Field(ge=0.0, le=1.0)
    gaps: List[str] = Field(default_factory=list)
    rubric: str = ""
    reasoning: str = ""


class QuorumResult(BaseModel):
    target: str
    passed: bool
    agreement_ratio: float
    composite_quality: float
    contested: bool
    verdicts: List[JudgeVerdict] = Field(default_factory=list)
    gap_list: List[str] = Field(default_factory=list)
    mean_score: float = 0.0


class IterationResult(BaseModel):
    iteration: int
    queries_used: List[str] = Field(default_factory=list)
    sources_found: int = 0
    new_sources: int = 0
    synthesis_excerpt: str = ""
    quorum_result: Optional[QuorumResult] = None
    quality: float = 0.0
    gap_list: List[str] = Field(default_factory=list)


class RouterOutput(BaseModel):
    category: QueryCategory
    secondary_category: Optional[QueryCategory] = None
    eligible_sources: List[str] = Field(default_factory=list)
    judge_rubric: str = ""
    sensitivity: bool = False
    reasoning: str = ""


class RunTrace(BaseModel):
    query: str
    category: QueryCategory
    sensitivity: bool
    mode: OperationMode
    iterations: List[IterationResult] = Field(default_factory=list)
    final_confidence: float = 0.0
    contested_positions: List[Dict[str, Any]] = Field(default_factory=list)
    total_sources: int = 0
    elapsed_seconds: float = 0.0


class QueryRequest(BaseModel):
    query: str
    mode: OperationMode = OperationMode.NORMAL
    depth: int = Field(default=3, ge=1, le=5)
    accuracy: int = Field(default=3, ge=1, le=5)


class SSEEvent(BaseModel):
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps({"type": self.type, **self.data})
        return f"data: {payload}\n\n"
