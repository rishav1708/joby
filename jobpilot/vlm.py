"""Read resumes and job posts from PDFs, scans or screenshots with Gemini's vision model into validated JSON.

    python -m jobpilot.vlm resume.pdf          # Profile
    python -m jobpilot.vlm post.png --job      # JobPost
"""
import base64
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .llm import llm


class Profile(BaseModel):
    name: str | None = None
    headline: str = Field(description="one-line summary of the candidate")
    skills: list[str] = Field(description="technical skills, tools, frameworks, languages")
    years_experience: float = Field(description="total professional/internship years, 0 if student")
    target_roles: list[str] = Field(description="job titles this person is a strong fit for")
    education: str | None = None
    locations: list[str] = Field(default_factory=list)


class JobPost(BaseModel):
    title: str
    company: str | None = None
    location: str | None = None
    required_skills: list[str]
    nice_to_have: list[str] = Field(default_factory=list)
    min_years: float | None = None


PROMPT = "Extract the {name} from this document. Only use what is visible; never invent skills."


MAGIC = {b"%PDF": "application/pdf", b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg",
         b"GIF8": "image/gif", b"RIFF": "image/webp"}


def file_block(path: Path) -> dict:
    """Detect PDF/image from the file's first bytes (extension can be missing or wrong)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path.resolve()} - check the path (quote it if it has spaces)")
    data = path.read_bytes()
    mime = next((m for sig, m in MAGIC.items() if data.startswith(sig)), None)
    if mime is None:
        raise ValueError(f"{path.name} is not a PDF, PNG, JPEG, GIF or WebP file "
                         "(for .docx or Pages, export it as PDF first)")
    return {"type": "file", "source_type": "base64", "mime_type": mime, "data": base64.b64encode(data).decode()}


def read(path: str | Path, schema: type[BaseModel] = Profile) -> BaseModel:
    """VLM path: the model sees the page pixels/PDF directly (works on scans and screenshots)."""
    msg = HumanMessage(content=[{"type": "text", "text": PROMPT.format(name=schema.__name__)},
                                file_block(Path(path))])
    return llm().with_structured_output(schema).invoke([msg])


def read_baseline(path: str | Path, schema: type[BaseModel] = Profile) -> BaseModel:
    """Baseline for evals: text layer (pypdf) or OCR (tesseract), then a text-only LLM call."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    else:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(path))
    prompt = PROMPT.format(name=schema.__name__) + "\n\nDOCUMENT:\n" + text
    return llm().with_structured_output(schema).invoke(prompt)


if __name__ == "__main__":
    schema = JobPost if "--job" in sys.argv else Profile
    print(read(sys.argv[1], schema).model_dump_json(indent=2))
