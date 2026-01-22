"""CLI entrypoint for ingestion."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Mapping

from oncology_rag.ingestion.chunking import build_chunker
from oncology_rag.ingestion.loaders.text_loader import load_markdown_documents
from oncology_rag.retrieval.embeddings import build_embedding_model
from oncology_rag.retrieval.vectorstores.chroma_store import ChromaStore


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(val) for key, val in value.items()}
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency wiring
        raise ImportError("PyYAML is required to load config files") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest markdown into ChromaDB.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Project config YAML.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Root directory containing markdown files.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        help="Glob pattern for markdown files.",
    )
    parser.add_argument(
        "--chunking-config",
        default=None,
        help="Chunking config YAML override.",
    )
    parser.add_argument(
        "--embeddings-config",
        default=None,
        help="Embeddings config YAML override.",
    )
    parser.add_argument(
        "--chroma-config",
        default=None,
        help="Chroma config YAML override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of markdown files to ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many chunks would be ingested.",
    )
    return parser


def _resolve_rag_paths(config_path: Path, project_cfg: Mapping[str, Any]) -> dict[str, Path]:
    rag_cfg = project_cfg.get("rag", {}) or {}
    base_dir = config_path.parent
    return {
        "chunking": _resolve_path(
            base_dir, str(rag_cfg.get("chunking", "configs/rag/chunking.yaml"))
        ),
        "embeddings": _resolve_path(
            base_dir, str(rag_cfg.get("embeddings", "configs/rag/embeddings.yaml"))
        ),
        "chroma": _resolve_path(
            base_dir, str(rag_cfg.get("chroma", "configs/rag/chroma.yaml"))
        ),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    project_cfg = _load_yaml(config_path)
    rag_paths = _resolve_rag_paths(config_path, project_cfg)

    chunking_path = Path(args.chunking_config) if args.chunking_config else rag_paths["chunking"]
    embeddings_path = (
        Path(args.embeddings_config) if args.embeddings_config else rag_paths["embeddings"]
    )
    chroma_path = Path(args.chroma_config) if args.chroma_config else rag_paths["chroma"]

    paths_cfg = project_cfg.get("paths", {}) or {}
    base_dir = config_path.parent
    default_input = _resolve_path(base_dir, str(paths_cfg.get("data_raw", "data/raw")))
    input_dir = Path(args.input).resolve() if args.input else default_input

    documents = load_markdown_documents(input_dir, pattern=args.pattern)
    if args.limit:
        documents = documents[: max(0, int(args.limit))]

    if not documents:
        print(f"No markdown files found in {input_dir}")
        return

    chunker = build_chunker(_load_yaml(chunking_path))
    total_chunks = sum(len(chunker.chunk(doc.text)) for doc in documents)

    if args.dry_run:
        print(f"Found {len(documents)} markdown files.")
        print(f"Would ingest {total_chunks} chunks.")
        return

    embeddings_cfg = _load_yaml(embeddings_path)
    chroma_cfg = _load_yaml(chroma_path)
    embedding_model = build_embedding_model(embeddings_cfg)
    store = ChromaStore(chroma_cfg)

    batch_size = int(args.batch_size or embeddings_cfg.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metadatas: list[Mapping[str, Any]] = []
    ingested_chunks = 0

    def flush_batch() -> None:
        nonlocal ingested_chunks
        if not batch_texts:
            return
        embeddings = embedding_model.embed_texts(batch_texts)
        store.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )
        ingested_chunks += len(batch_texts)
        batch_ids.clear()
        batch_texts.clear()
        batch_metadatas.clear()

    for doc in documents:
        chunks = chunker.chunk(doc.text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc.doc_id}::chunk_{idx:04d}"
            metadata: dict[str, Any] = {
                "source": doc.doc_id,
                "source_path": doc.source_path,
                "chunk_index": idx,
                "token_start": chunk.start,
                "token_end": chunk.end,
            }
            if doc.metadata:
                metadata.update(doc.metadata)
            if doc.title:
                metadata["title"] = doc.title
            if chunk.metadata:
                metadata.update(chunk.metadata)
            batch_ids.append(chunk_id)
            batch_texts.append(chunk.text)
            batch_metadatas.append(metadata)
            if len(batch_texts) >= batch_size:
                flush_batch()
        flush_batch()

    print(f"Ingested {ingested_chunks} chunks into ChromaDB.")


if __name__ == "__main__":
    main()
