"""Gemini clients. Free tier works; override models with env vars when Google renames them."""
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

LLM_MODEL = os.getenv("JOBPILOT_LLM", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("JOBPILOT_EMBED", "gemini-embedding-001")


def llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    # max_retries rides out free-tier 429s with backoff
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature, max_retries=6)


def embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
