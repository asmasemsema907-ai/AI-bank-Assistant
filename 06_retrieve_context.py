from importlib import import_module
from typing import Any

hybrid_search = import_module("04_vector_representation").hybrid_search


def _is_current(row: dict[str, Any]) -> bool:
    value = row.get("is_current")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def build_context(
    question: str,
    k: int = 4,
    max_sources: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    if not question or not question.strip():
        return "", []
    if k <= 0 or max_sources <= 0:
        return "", []

    rows = hybrid_search(question.strip(), k=k)
    rows = sorted(rows, key=lambda row: (_is_current(row), row["score"]), reverse=True)

    selected = []
    seen_documents = set()

    for row in rows:
        if row["score"] <= 0:
            continue
        if row["document_id"] in seen_documents:
            continue
        selected.append(row)
        seen_documents.add(row["document_id"])
        if len(selected) == max_sources:
            break

    context_parts = []
    for source_number, row in enumerate(selected, start=1):
        status = "CURRENT" if _is_current(row) else "OUTDATED"
        context_parts.append(
            f"[Source {source_number}] {row['title']} ({status})\n{row['chunk_text']}"
        )

    return "\n\n".join(context_parts), selected
