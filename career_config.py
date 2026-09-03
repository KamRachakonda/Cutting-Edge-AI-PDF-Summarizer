import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY","").strip()
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY","").strip()
MODEL=os.getenv("MODEL","openai/gpt-oss-20b")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2")
CHROMA_DIR=os.getenv("CHROMA_DIR","./data/chroma")
