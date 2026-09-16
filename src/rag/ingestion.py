"""Build a deterministic local Chroma index from the PH9-01 policy corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import RagSettings
from src.rag.corpus_validation import load_manifest, validate_corpus


class CorpusIndexError(RuntimeError):
    """Raised when the approved corpus cannot be indexed safely."""


@dataclass(frozen=True)
class IndexSummary:
    """Small, reviewable result from one repeatable index build."""

    collection_name: str
    document_count: int
    section_count: int
    chunk_count: int
    source_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]


def _section_text(document: str, section_id: str, heading: str) -> str:
    marker = f"## {section_id} - {heading}"
    if marker not in document:
        raise CorpusIndexError(f"Declared section is missing: {section_id}")
    content = document.split(marker, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return content.strip()


def _chunk_id(metadata: dict[str, Any], content: str) -> str:
    """Keep unchanged chunks stable across machines and repeated rebuilds."""

    identity = {
        "source_id": metadata["source_id"],
        "section_id": metadata["section_id"],
        "chunk_index": metadata["chunk_index"],
        "content": content,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"{metadata['source_id']}:{metadata['section_id']}:{digest[:20]}"


def create_policy_chunks(settings: RagSettings) -> tuple[list[Document], list[str]]:
    """Split within declared sections so every future citation stays meaningful."""

    violations = validate_corpus(settings.corpus_path)
    if violations:
        raise CorpusIndexError("Corpus validation failed: " + "; ".join(violations))

    manifest = load_manifest(settings.corpus_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks: list[Document] = []
    chunk_ids: list[str] = []

    for document_metadata in manifest["documents"]:
        source_path = settings.corpus_path / document_metadata["path"]
        source_text = source_path.read_text(encoding="utf-8")
        for section in document_metadata["sections"]:
            section_text = _section_text(
                source_text,
                section["section_id"],
                section["heading"],
            )
            section_chunks = splitter.split_text(section_text)
            for chunk_index, content in enumerate(section_chunks):
                metadata = {
                    "corpus_id": manifest["corpus_id"],
                    "source_id": document_metadata["source_id"],
                    "title": document_metadata["title"],
                    "version": document_metadata["version"],
                    "source_path": document_metadata["path"],
                    "section_id": section["section_id"],
                    "section_heading": section["heading"],
                    "chunk_index": chunk_index,
                    "fictional": True,
                }
                chunk_id = _chunk_id(metadata, content)
                metadata["chunk_id"] = chunk_id
                chunks.append(Document(page_content=content, metadata=metadata))
                chunk_ids.append(chunk_id)

    if len(chunk_ids) != len(set(chunk_ids)):
        raise CorpusIndexError("Deterministic chunk IDs must be unique")
    return chunks, chunk_ids


def create_embedding_model(settings: RagSettings) -> HuggingFaceEmbeddings:
    """Load a local sentence-transformer; no hosted inference API is used."""

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def open_vector_store(settings: RagSettings, embedding_function: Any) -> Chroma:
    """Open the configured local cosine-similarity collection."""

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_function,
        persist_directory=str(settings.vector_store_path),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def rebuild_index(
    settings: RagSettings,
    embedding_function: Any | None = None,
) -> IndexSummary:
    """Replace generated collection contents with one deterministic corpus build."""

    chunks, chunk_ids = create_policy_chunks(settings)
    try:
        embedding = embedding_function or create_embedding_model(settings)
        settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        store = open_vector_store(settings, embedding)

        # Removing known generated rows first also removes chunks deleted from source.
        existing_ids = store.get(include=[]).get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_documents(chunks, ids=chunk_ids)
    except Exception as error:
        raise CorpusIndexError("The local policy index could not be rebuilt") from error

    manifest = load_manifest(settings.corpus_path)
    return IndexSummary(
        collection_name=settings.collection_name,
        document_count=len(manifest["documents"]),
        section_count=sum(len(item["sections"]) for item in manifest["documents"]),
        chunk_count=len(chunks),
        source_ids=tuple(item["source_id"] for item in manifest["documents"]),
        chunk_ids=tuple(chunk_ids),
    )
