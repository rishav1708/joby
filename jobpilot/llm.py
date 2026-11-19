"""Model clients. Chat/VLM on the Gemini free tier; embeddings run locally so indexing never hits API quota.
Override models with env vars (JOBPILOT_LLM, JOBPILOT_EMBED)."""
import os
from functools import cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

LLM_MODEL = os.getenv("JOBPILOT_LLM", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("JOBPILOT_EMBED", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval instruction


def llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    # Extraction/grading don't need deep reasoning: low thinking cuts latency a lot on Gemini 3+.
    # Few retries + a request timeout so a free-tier 429 surfaces in seconds instead of minutes of backoff.
    thinking = {} if LLM_MODEL.startswith("gemini-2") else {"thinking_level": os.getenv("JOBPILOT_THINKING", "low")}
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature, max_retries=2, timeout=60, **thinking)


class LocalEmbeddings(Embeddings):
    """sentence-transformers behind the LangChain Embeddings interface (so RAGAS can use it too)."""

    def __init__(self, model: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy: pulls in torch
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
        return self.model.encode(prefix + text, normalize_embeddings=True).tolist()


@cache
def embedder() -> LocalEmbeddings:
    return LocalEmbeddings()
