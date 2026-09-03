"""Deterministic, explainable career-intelligence primitives."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from html import unescape
from typing import Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    WEB_TERMS = frozenset({"current", "latest", "recent", "today", "company", "competitor", "salary", "market", "industry", "news", "culture", "stock", "hiring"})
    def route(self, query: str, has_documents: bool = True, web_enabled: bool = True) -> QueryRoute:
        terms = set(re.findall(r"[a-z]+", query.lower()))
        needs_web = bool(terms & self.WEB_TERMS)
        if needs_web and has_documents and web_enabled: return QueryRoute.HYBRID
        if needs_web and web_enabled: return QueryRoute.WEB
        return QueryRoute.DOCUMENT if has_documents else QueryRoute.WEB
    def explain(self, query: str, route: QueryRoute) -> str:
        return {QueryRoute.DOCUMENT: "Uses uploaded evidence only.", QueryRoute.WEB: "Uses external sources because the question needs current context.", QueryRoute.HYBRID: "Combines uploaded evidence with external context."}[route]

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
    def __init__(self, chunks: Sequence[DocumentChunk]): self.chunks = list(chunks)
    def retrieve(self, query: str, limit: int = 5) -> List[Evidence]:
        terms = set(re.findall(r"[a-z0-9+#.-]+", query.lower()))
        ranked = []
        for chunk in self.chunks:
            words = set(re.findall(r"[a-z0-9+#.-]+", chunk.text.lower()))
            overlap = len(terms & words)
            if overlap: ranked.append(Evidence(chunk.text, chunk.source, chunk.page, overlap / max(1, len(terms))))
        ranked.sort(key=lambda x: (-x.relevance, x.source, x.page or 0))
        return ranked[:limit]

class WebResearcher:
    def fetch(self, url: str, timeout: int = 10) -> Evidence:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc: raise ValueError("Research URLs must use http:// or https://.")
        request = Request(url, headers={"User-Agent": "CareerIntelligence/1.0"})
        with urlopen(request, timeout=timeout) as response: raw = response.read(500_000).decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", raw, flags=re.I|re.S)
        text = re.sub(r"<[^>]+>", " ", unescape(text)); text = re.sub(r"\s+", " ", text).strip()
        if not text: raise ValueError("The research source contained no readable text.")
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
    caveat: str = "Scores are deterministic and based on explicit evidence; absence of a term is not proof of lack of capability."
    @property
    def evidence(self) -> List[str]: return [e for d in self.dimensions for e in d.evidence]

class ExplainableResumeScorer:
    WEIGHTS = {"Technical & Architecture": .20, "Leadership": .15, "Customer & Consulting": .15, "Cloud": .10, "AI / GenAI": .15, "Business Impact": .15, "Industry Alignment": .10}
    VOCABULARY = {
        "Technical & Architecture": ["architecture", "solution architecture", "api", "integration", "system design", "software", "saas", "technical"],
        "Leadership": ["leadership", "leader", "manage", "managed", "mentor", "coach", "team", "director", "strategy"],
        "Customer & Consulting": ["customer", "client", "consulting", "consultant", "stakeholder", "trusted advisor", "discovery", "workshop"],
        "Cloud": ["cloud", "aws", "azure", "gcp", "kubernetes", "docker", "cloud architecture"],
        "AI / GenAI": ["ai", "genai", "generative ai", "machine learning", "llm", "agentic", "artificial intelligence"],
        "Business Impact": ["revenue", "growth", "roi", "business impact", "cost", "efficiency", "transformation", "kpi", "target"],
        "Industry Alignment": ["banking", "financial services", "consulting", "technology", "enterprise", "healthcare", "retail", "government"],
    }
    @staticmethod
    def _present(term: str, text: str) -> bool: return bool(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text.lower()))
    @staticmethod
    def _evidence(term: str, text: str) -> str:
        for line in text.splitlines():
            if ExplainableResumeScorer._present(term, line): return f"{term}: {line.strip()}"
        return f"{term}: explicit match found in resume."
    def score(self, resume_text: str, jd_text: str) -> ResumeJDScore:
        dimensions = []
        for name, weight in self.WEIGHTS.items():
            terms = [t for t in self.VOCABULARY[name] if self._present(t, jd_text)]
            if not terms: terms = [t for t in self.VOCABULARY[name] if self._present(t, resume_text)]
            matched = tuple(t for t in terms if self._present(t, resume_text)); missing = tuple(t for t in terms if t not in matched)
            dimensions.append(ScoreDimension(name, round(100*len(matched)/max(1,len(terms)),2), matched, missing, tuple(self._evidence(t,resume_text) for t in matched)))
        return ResumeJDScore(round(sum(d.score*self.WEIGHTS[d.name] for d in dimensions),2), tuple(dimensions))

def grounded_prompt(task: str, evidence: Iterable[Evidence | str]) -> str:
    lines = [f"- [{x.citation}] {x.text}" if isinstance(x,Evidence) else f"- {x}" for x in evidence]
    return "Use only the evidence below. Never invent employers, dates, degrees, certifications, skills, metrics, or achievements. Mark unsupported details as [NEEDS CONFIRMATION]. Cite evidence for material claims.\n\nTask: " + task + "\n\nEvidence:\n" + "\n".join(lines)

def career_report_markdown(score: ResumeJDScore, recommendations: Sequence[str], research: Sequence[Evidence] = ()) -> str:
    lines=["# Career Intelligence Report","",f"**Resume-JD score:** {score.overall}/100","","## Score Breakdown"]
    for d in score.dimensions:
        lines.append(f"- **{d.name}:** {d.score}/100")
        if d.missing: lines.append(f"  - Missing explicit evidence: {', '.join(d.missing)}")
    lines += ["","## Recommendations"]; lines += [f"- {r}" for r in recommendations]; lines += ["","## Evidence"]; lines += [f"- {e}" for e in score.evidence]
    if research: lines += ["","## External Research"] + [f"- [{e.citation}] {e.text[:1200]}" for e in research]
    lines += ["",f"_{score.caveat}_"]
    return "\n".join(lines)
