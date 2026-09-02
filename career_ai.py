import json
from typing import Any, Dict, List
from groq import Groq
from career_config import GROQ_API_KEY, MODEL, WEIGHTS

class CareerAI:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")
        self.client = Groq(api_key=GROQ_API_KEY)

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        r = self.client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""

    def json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.chat(system + " Return valid JSON only. No markdown fences.", user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
            raise ValueError("The model returned invalid JSON")

    def analyze_resume(self, text: str) -> Dict[str, Any]:
        return self.json("""You are an expert executive recruiter and technology career strategist. Extract only facts supported by the resume. Never invent experience.""", f"""Analyze this resume and return keys: executive_profile, years_experience, industries, companies, roles, technical_skills, cloud_skills, ai_skills, leadership, customer_experience, architecture, business_impact, achievements, certifications, differentiators. Values should be concise lists or strings.\n\nRESUME:\n{text[:24000]}""")

    def analyze_pd(self, text: str) -> Dict[str, Any]:
        return self.json("""You are an expert executive recruiter. Convert a position/job description into a structured competency model.""", f"""Return keys: role_title, company, responsibilities, required_skills, preferred_skills, technical_requirements, leadership_requirements, customer_requirements, business_expectations, success_metrics, ai_requirements, cloud_requirements, industry_requirements.\n\nPOSITION DESCRIPTION:\n{text[:24000]}""")

    def match(self, resume: Dict, pd: Dict) -> Dict[str, Any]:
        return self.json("""You are a senior hiring strategist. Compare candidate evidence to role requirements. Do not assume an unmentioned skill exists. Score each dimension 0-100 and provide evidence.""", f"""Candidate:\n{json.dumps(resume, ensure_ascii=False)}\n\nRole:\n{json.dumps(pd, ensure_ascii=False)}\n\nReturn JSON with dimensions (Technical & Architecture, Leadership, Customer & Consulting, Cloud, AI / GenAI, Business Impact, Industry Alignment). For each include score, status (Strong/Partial/Gap), evidence, missing, recommendation. Also include overall_rationale and top_priorities.\nWeighted scoring weights are {json.dumps(WEIGHTS)}; do not fabricate an overall score.""")

    def advisor(self, question: str, evidence: str, research: str = "") -> str:
        return self.chat("""You are an evidence-grounded AI career advisor. Candidate documents are authoritative for candidate facts. External research is context only. Never invent candidate experience. Clearly distinguish evidence, inference and recommendation. Give practical executive-level advice.""", f"QUESTION:\n{question}\n\nCANDIDATE/ROLE EVIDENCE:\n{evidence}\n\nEXTERNAL RESEARCH:\n{research or 'None'}")

    def generate(self, task: str, source: str) -> str:
        return self.chat("""You are an executive resume and career communications specialist. Preserve factual accuracy. Use strong achievement-oriented language without inventing metrics, employers, technologies or responsibilities.""", f"TASK:\n{task}\n\nSOURCE MATERIAL:\n{source}", 0.2)
