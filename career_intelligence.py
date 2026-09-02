"""Core career-intelligence primitives that do not depend on Streamlit or an LLM."""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        location = f", page {self.page}" if self.page else ""
        section = f", {self.section}" if self.section else ""
        return f"{self.source}{location}{section}"


class QueryRoute(str, Enum):
    DOCUMENT = "document"
    WEB = "web"
    HYBRID = "hybrid"


class QueryRouter:
    """Route questions using explicit source availability and intent signals."""

    WEB_TERMS = frozenset({
        "current", "latest", "recent", "today", "company", "competitor",
        "salary", "market", "industry", "news", "linkedin", "website",
        "role", "job", "hiring", "culture",
    })

    def route(self, query: str, has_documents: bool = True,
              web_enabled: bool = True) -> QueryRoute:
        normalized = set(re.findall(r"[a-z]+", query.lower()))
        requests_web = bool(normalized & self.WEB_TERMS)
        if requests_web and has_documents and web_enabled:
            return QueryRoute.HYBRID
        if requests_web and web_enabled:
            return QueryRoute.WEB
        return QueryRoute.DOCUMENT if has_documents else QueryRoute.WEB

    def explain(self, query: str, route: QueryRoute) -> str:
        if route is QueryRoute.DOCUMENT:
            return "Answered from uploaded documents because no current web source was requested."
        if route is QueryRoute.WEB:
            return "Answered from web research because the question requires current external information."
        return "Combined uploaded-document evidence with current web research."


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str
    page: Optional[int] = None
    relevance: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}" if self.page else self.source


class SourceAwareRetriever:
    """Small deterministic lexical retriever that retains source/page provenance."""

    def __init__(self, chunks: Sequence[DocumentChunk]):
        self.chunks = list(chunks)

    def retrieve(self, query: str, limit: int = 5) -> List[Evidence]:
        terms = set(re.findall(r"[a-z0-9+#.-]+", query.lower()))
        ranked = []
        for chunk in self.chunks:
            words = set(re.findall(r"[a-z0-9+#.-]+", chunk.text.lower()))
            overlap = len(terms & words)
            if overlap:
                ranked.append(Evidence(chunk.text, chunk.source, chunk.page,
                                       overlap / max(1, len(terms))))
        ranked.sort(key=lambda item: (-item.relevance,
                    item.source, item.page or 0))
        return ranked[:limit]


class WebResearcher:
    """Fetch user-supplied public sources without inventing company facts."""

    def fetch(self, url: str, timeout: int = 10) -> Evidence:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Research URLs must use http:// or https://.")
        request = Request(
            url, headers={"User-Agent": "CareerIntelligence/1.0"})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(500_000).decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", raw,
                      flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("The research source contained no readable text.")
        return Evidence(text[:12000], url)


@dataclass(frozen=True)
class ScoreDimension:
    name: str
    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ResumeJDScore:
    overall: float
    dimensions: tuple[ScoreDimension, ...]
    caveat: str = "Score reflects only explicit text matches; it does not infer experience."

    @property
    def evidence(self) -> List[str]:
        return [item for dimension in self.dimensions for item in dimension.evidence]


class ExplainableResumeScorer:
    """Deterministic score based on explicit terms and resume evidence only."""

    WEIGHTS = {"skills": 0.50, "experience": 0.30, "education": 0.20}
    SECTION_TERMS = {
        "skills": ("requirements", "skills", "technologies", "tools", "qualifications"),
        "experience": ("experience", "responsibilities", "years", "senior", "lead"),
        "education": ("education", "degree", "certification", "bachelor", "master"),
    }

    def score(self, resume_text: str, jd_text: str) -> ResumeJDScore:
        resume = resume_text.lower()
        jd = jd_text.lower()
        dimensions = []
        for name, weight in self.WEIGHTS.items():
            candidates = self._terms_for_dimension(jd, name)
            matched = tuple(
                term for term in candidates if self._term_in_text(term, resume))
            missing = tuple(term for term in candidates if term not in matched)
            score = round(100 * len(matched) / max(1, len(candidates)), 2)
            evidence = tuple(self._evidence_for(term, resume_text)
                             for term in matched)
            dimensions.append(ScoreDimension(
                name, score, matched, missing, evidence))
        overall = round(
            sum(item.score * self.WEIGHTS[item.name] for item in dimensions), 2)
        return ResumeJDScore(overall, tuple(dimensions))

    def _terms_for_dimension(self, jd: str, dimension: str) -> List[str]:
        sentences = re.split(r"[\n.;]+", jd)
        selected = " ".join(sentence for sentence in sentences
                            if any(marker in sentence for marker in self.SECTION_TERMS[dimension]))
        terms = re.findall(r"[a-z][a-z0-9+#.-]{2,}", selected)
        stop = {
            "the", "and", "with", "for", "from", "this", "that", "are", "you", "our",
            "skills", "requirements", "technologies", "tools", "qualifications",
            "experience", "responsibilities", "years", "senior", "lead",
            "education", "degree", "certification", "bachelor", "master",
        }
        return list(dict.fromkeys(term for term in terms if term not in stop))

    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))

    @staticmethod
    def _evidence_for(term: str, resume_text: str) -> str:
        for line in resume_text.splitlines():
            if ExplainableResumeScorer._term_in_text(term, line.lower()):
                return f"Resume evidence for '{term}': {line.strip()}"
        return f"Resume evidence for '{term}': explicit match found."


def grounded_prompt(task: str, evidence: Iterable[Evidence | str]) -> str:
    """Build a prompt that forbids unsupported career claims."""
    evidence_lines = []
    for item in evidence:
        if isinstance(item, Evidence):
            evidence_lines.append(f"- [{item.citation}] {item.text}")
        else:
            evidence_lines.append(f"- {item}")
    return (
        "You are a careful career assistant. Use only the evidence below. "
        "Never invent employers, dates, degrees, certifications, skills, metrics, "
        "or achievements. Mark missing information as [NEEDS CONFIRMATION]. "
        "Cite the evidence supporting every major recommendation.\n\n"
        f"Task: {task}\n\nEvidence:\n" + "\n".join(evidence_lines)
    )


def career_report_markdown(score: ResumeJDScore, recommendations: Sequence[str],
                           research: Sequence[Evidence] = ()) -> str:
    """Create a portable career-intelligence report suitable for download."""
    lines = ["# Career Intelligence Report", "", f"**Resume-JD score:** {score.overall}/100", "",
             "## Score Breakdown"]
    for dimension in score.dimensions:
        lines.append(f"- **{dimension.name.title()}:** {dimension.score}/100")
        if dimension.missing:
            lines.append(
                f"  Missing explicit terms: {', '.join(dimension.missing)}")
    lines.extend(["", "## Recommendations"])
    lines.extend(f"- {recommendation}" for recommendation in recommendations)
    lines.extend(["", "## Evidence"])
    lines.extend(f"- {evidence}" for evidence in score.evidence)
    lines.extend(f"- [{item.citation}] {item.text}" for item in research)
    lines.append(f"\n_{score.caveat}_")
    return "\n".join(lines)
