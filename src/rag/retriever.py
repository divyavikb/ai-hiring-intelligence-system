
from __future__ import annotations

from pathlib import Path
import re

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "hiring_knowledge"


def load_knowledge_documents() -> list[dict]:
    """Load Markdown knowledge-base files and split them by section."""

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: "
            f"{KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        current_heading = "Introduction"
        section_lines = []

        def save_section() -> None:
            text = "\n".join(section_lines).strip()

            if text:
                documents.append(
                    {
                        "id": (
                            f"{path.stem}:"
                            f"{current_heading.lower().replace(' ', '_')}"
                        ),
                        "text": text,
                        "source": path.name,
                        "agent":path.stem,
                        "section": current_heading,
                    }
                )

        for line in content.splitlines():
            heading_match = re.match(r"^#{1,6}\s+(.+)$", line)

            if heading_match:
                save_section()
                current_heading = heading_match.group(1).strip()
                section_lines = []
            else:
                section_lines.append(line)

        save_section()

    if not documents:
        raise ValueError(
            f"No non-empty Markdown sections found in "
            f"{KNOWLEDGE_BASE_DIR}"
        )

    return documents


def get_collection():
    """Return the persistent ChromaDB collection."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"description": "Hiring guidelines and benchmarks"},
    )


def index_knowledge_base() -> int:
    """Index the Markdown sections, updating existing documents."""

    documents = load_knowledge_documents()
    collection = get_collection()

    collection.upsert(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[
            {
                "source": doc["source"],
                "section": doc["section"],
                "agent": doc["agent"],
            }
            for doc in documents
        ],
    )

    return len(documents)


def retrieve_context(
    query: str,
    agents =None,
    n_results: int = 3,
) -> list[dict]:
    """Retrieve relevant knowledge-base sections for a query."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    collection = get_collection()

    if collection.count() == 0:
        index_knowledge_base()

        # Reopen after indexing to access the populated collection.
        collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where={"agent": {"$in": agents}} if agents else None,

    )

    retrieved = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] or {}

        retrieved.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
            }
        )

    return retrieved
