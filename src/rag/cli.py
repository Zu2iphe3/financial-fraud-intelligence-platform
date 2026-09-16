"""Small command-line interface for building and reviewing the local RAG flow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.rag.assistant import GroundingError, OllamaUnavailableError, PolicyAssistant
from src.rag.config import DEFAULT_CONFIG_PATH, RagConfigurationError, load_rag_settings
from src.rag.ingestion import CorpusIndexError, rebuild_index
from src.rag.retrieval import PolicyRetriever, RetrievalError, evaluate_retrieval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local fictional-policy RAG assistant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="rebuild the local Chroma policy index")

    retrieve = commands.add_parser("retrieve", help="show relevant policy evidence")
    retrieve.add_argument("question")

    ask = commands.add_parser("ask", help="answer from retrieved policy evidence")
    ask.add_argument("question")

    commands.add_parser("evaluate", help="run the small relevance-threshold review")
    return parser


def _run(args: argparse.Namespace) -> int:
    settings = load_rag_settings(args.config)

    if args.command == "index":
        print(json.dumps(asdict(rebuild_index(settings)), indent=2))
        return 0

    retriever = PolicyRetriever.from_settings(settings)
    if args.command == "retrieve":
        print(json.dumps([asdict(item) for item in retriever.retrieve(args.question)], indent=2))
        return 0
    if args.command == "evaluate":
        print(json.dumps(evaluate_retrieval(retriever).to_dict(), indent=2))
        return 0

    result = PolicyAssistant(settings, retriever).answer(args.question)
    print(json.dumps(asdict(result), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (
        CorpusIndexError,
        GroundingError,
        OllamaUnavailableError,
        RagConfigurationError,
        RetrievalError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
