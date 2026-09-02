import uuid
from typing import List, Dict
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from career_config import CHROMA_DIR, EMBEDDING_MODEL
from career_documents import chunk_text

class CareerRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.name = "career_" + uuid.uuid4().hex[:12]
        self.embedding = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(self.name, embedding_function=self.embedding)

    def add_documents(self, documents: List[Dict]):
        ids, texts, metas = [], [], []
        for doc in documents:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                ids.append(uuid.uuid4().hex)
                texts.append(chunk)
                metas.append({"filename": doc["filename"], "type": doc["type"], "chunk": i + 1})
        if texts:
            self.collection.add(ids=ids, documents=texts, metadatas=metas)

    def search(self, query: str, n: int = 6) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_texts=[query], n_results=min(n, self.collection.count()))
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [{"text": d, **(m or {})} for d, m in zip(docs, metas)]

    def context(self, query: str, n: int = 8) -> str:
        hits = self.search(query, n)
        return "\n\n".join(f"[{h.get('filename')} | {h.get('type')} | chunk {h.get('chunk')}]\n{h['text']}" for h in hits)
