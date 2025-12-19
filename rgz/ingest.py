import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt

from config import (
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_RAW_DIR,
    EMBED_MODEL,
)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    return docx2txt.process(str(path))


def load_documents() -> List[Tuple[str, dict]]:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for ext in ("*.txt", "*.md", "*.pdf", "*.docx"):
        for file_path in DATA_RAW_DIR.rglob(ext):
            text = ""
            if file_path.suffix.lower() in {".txt", ".md"}:
                text = read_txt(file_path)
            elif file_path.suffix.lower() == ".pdf":
                text = read_pdf(file_path)
            elif file_path.suffix.lower() == ".docx":
                text = read_docx(file_path)

            if not text.strip():
                continue

            rel = file_path.relative_to(BASE_DIR)
            docs.append(
                (
                    text,
                    {
                        "source": str(rel),
                        "filename": file_path.name,
                    },
                )
            )
    return docs


def build_chunks(docs: List[Tuple[str, dict]]) -> Tuple[List[str], List[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts: List[str] = []
    metadatas: List[dict] = []

    for doc_text, meta in docs:
        for idx, chunk in enumerate(splitter.split_text(doc_text)):
            texts.append(chunk)
            metadatas.append({**meta, "chunk": idx})
    return texts, metadatas


def ensure_collection(client: chromadb.Client, reset: bool):
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    try:
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def upsert_chunks(texts: List[str], metadatas: List[dict], reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = ensure_collection(client, reset)

    ids = []
    for idx, meta in enumerate(metadatas):
        raw_id = f"{meta.get('source','unknown')}::{meta.get('chunk', idx)}::{idx}"
        ids.append(hashlib.md5(raw_id.encode("utf-8")).hexdigest())

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embed_fn(texts),
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest educational data into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing collection before ingesting.",
    )
    args = parser.parse_args()

    docs = load_documents()
    if not docs:
        print("No documents found in data/raw. Add educational materials and retry.")
        return

    texts, metas = build_chunks(docs)
    if not texts:
        print("No chunks produced.")
        return

    upsert_chunks(texts, metas, reset=args.reset)
    print(f"Ingested {len(texts)} educational chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()

