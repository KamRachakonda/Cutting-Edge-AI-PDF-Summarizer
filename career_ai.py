import json
from typing import Any, Callable, Dict, Optional
from groq import Groq
from career_config import GROQ_API_KEY, MODEL


class CareerAI:
    """LLM layer: explanations and generation only; scoring remains deterministic."""

    def __init__(self, usage_callback: Optional[Callable[[Dict[str, int]], None]] = None):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.usage_callback = usage_callback

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=MODEL, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        usage = getattr(response, "usage", None)
        if usage is not None and self.usage_callback:
            self.usage_callback({
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "requests": 1,
            })
        return response.choices[0].message.content or ""

    def json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.chat(
            system + " Return valid JSON only. No markdown fences.", user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end+1])
            raise ValueError("The model returned invalid JSON")

    def analyze_resume(self, text: str) -> Dict[str, Any]:
        return self.json("You are an executive recruiter. Extract only facts explicitly supported by the resume. Never invent experience.",
                         "Return concise keys: executive_profile, years_experience, industries, companies, roles, technical_skills, cloud_skills, ai_skills, leadership, customer_experience, architecture, business_impact, achievements, certifications, differentiators.\n\nRESUME:\n" + text[:30000])

    def analyze_pd(self, text: str) -> Dict[str, Any]:
        return self.json("You are an executive recruiter. Convert a job description into a structured competency model.",
                         "Return keys: role_title, company, responsibilities, required_skills, preferred_skills, technical_requirements, leadership_requirements, customer_requirements, business_expectations, success_metrics, ai_requirements, cloud_requirements, industry_requirements.\n\nJOB DESCRIPTION:\n" + text[:30000])

    def advisor(self, question: str, evidence: str, research: str = "") -> str:
        return self.chat("You are an evidence-grounded career advisor. Uploaded documents are authoritative for candidate facts; external research is context only. Never invent candidate experience. Distinguish evidence, inference and recommendation. Include source labels when provided.",
                         f"QUESTION:\n{question}\n\nDOCUMENT EVIDENCE:\n{evidence}\n\nEXTERNAL RESEARCH:\n{research or 'None'}")

    def generate(self, task: str, source: str) -> str:
        return self.chat("You are an executive career communications specialist. Preserve factual accuracy. Never invent metrics, employers, dates, technologies or responsibilities. Mark unsupported facts [NEEDS CONFIRMATION].",
                         f"TASK:\n{task}\n\nSOURCE MATERIAL:\n{source}", 0.2)
