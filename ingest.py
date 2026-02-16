import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _chunk_tokens(tokens: List[str], chunk_size: int, chunk_overlap: int) -> Iterable[List[str]]:
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if chunk:
            yield chunk


def _extract_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    if PdfReader is None:
        raise RuntimeError("pypdf is required for PDF ingestion. Install with `pip install pypdf`.")
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _normalize_whitespace(text)
        if text:
            pages.append((i, text))
    return pages


def _extract_txt_pages(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = _normalize_whitespace(text)
    return [(1, text)] if text else []


def _discover_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files: List[Path] = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in {".pdf", ".txt"}:
                files.append(Path(root) / name)
    return sorted(files)


def ingest(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> int:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    files = _discover_files(input_path)
    if not files:
        raise FileNotFoundError("No .pdf or .txt files found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as f:
        for path in files:
            ext = path.suffix.lower()
            if ext == ".pdf":
                pages = _extract_pdf_pages(path)
            elif ext == ".txt":
                pages = _extract_txt_pages(path)
            else:
                continue

            chunk_index = 0
            for page_number, page_text in pages:
                tokens = _tokenize(page_text)
                for chunk_tokens in _chunk_tokens(tokens, chunk_size, chunk_overlap):
                    chunk_text = " ".join(chunk_tokens)
                    record = {
                        "text": chunk_text,
                        "metadata": {
                            "filename": path.name,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    chunk_index += 1
                    total_chunks += 1

    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="StudyBuddy ingestion pipeline.")
    parser.add_argument("--input", required=True, help="File or directory path containing PDF/TXT.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "ingested.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in tokens.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap in tokens.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    total = ingest(
        input_path=input_path,
        output_path=output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingestion complete. Chunks written: {total}. Output: {output_path}")


if __name__ == "__main__":
    main()