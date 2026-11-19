"""Model clients. Chat/VLM on the Gemini free tier; embeddings run locally so indexing never hits API quota.
Override models with env vars (JOBPILOT_LLM, JOBPILOT_LLM_FALLBACK, JOBPILOT_EMBED)."""
import os
from functools import cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

LLM_MODEL = os.getenv("JOBPILOT_LLM", "gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("JOBPILOT_LLM_FALLBACK", "gemini-3.5-flash-lite")  # "" disables
EMBED_MODEL = os.getenv("JOBPILOT_EMBED", "BAAI/bge-small-en-v1.5")
EMBED_MAX_CHARS = 1200  # ~256 tokens: title + first paragraphs carry the signal, and it's ~2x faster than 512
EMBED_KEY = f"fastembed:{EMBED_MODEL}@{EMBED_MAX_CHARS}"  # stored with cached vectors; a change re-embeds
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval instruction


def _gemini(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    # Low thinking: extraction/grading don't need deep reasoning and it cuts latency a lot on Gemini 3+.
    # max_retries=3 -> 2 SDK retries with backoff on 429/5xx, then we fall back instead of waiting minutes.
    thinking = {} if model.startswith("gemini-2") else {"thinking_level": os.getenv("JOBPILOT_THINKING", "low")}
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=3, timeout=60, **thinking)


def llm(temperature: float = 0.0):
    """Primary Gemini model, falling back to a lighter free model on 503 overload / 429 quota.
    Supports .with_structured_output() and .bind_tools() like a plain chat model."""
    primary = _gemini(LLM_MODEL, temperature)
    if not FALLBACK_MODEL or FALLBACK_MODEL == LLM_MODEL:
        return primary
    return primary.with_fallbacks([_gemini(FALLBACK_MODEL, temperature)])


class LocalEmbeddings(Embeddings):
    """ONNX (fastembed) behind the LangChain Embeddings interface, so RAGAS can use it too.
    No torch: the whole app fits in a 512MB free instance."""

    def __init__(self, model: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed([t[:EMBED_MAX_CHARS] for t in texts], batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        prefix = QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
        return next(iter(self.model.embed([prefix + text]))).tolist()


@cache
def embedder() -> LocalEmbeddings:
    return LocalEmbeddings()
