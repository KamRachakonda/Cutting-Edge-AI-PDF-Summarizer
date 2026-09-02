import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")

WEIGHTS = {
    "Technical & Architecture": 0.20,
    "Leadership": 0.15,
    "Customer & Consulting": 0.15,
    "Cloud": 0.10,
    "AI / GenAI": 0.15,
    "Business Impact": 0.15,
    "Industry Alignment": 0.10,
}
