import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    chunk_id: str
    section: str
    content: str

def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def create_chunk_id(section: str) -> str:
    normalized = unicodedata.normalize("NFKD", section)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowercased_text = ascii_text.lower()

    return re.sub(r"[^a-z0-9]+", "-", lowercased_text).strip("-")


def split_by_headings(markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                content = "\n".join(current_lines).strip()

                if content:
                    chunks.append(
                        Chunk(
                            chunk_id=create_chunk_id(current_section),
                            section=current_section,
                            content=content,
                        )
                    )

            current_section = line.removeprefix("## ").strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        content = "\n".join(current_lines).strip()

        if content:
            chunks.append(
                Chunk(
                    chunk_id=create_chunk_id(current_section),
                    section=current_section,
                    content=content
                )
            )

    return chunks