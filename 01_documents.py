"""
01_documents.py

Same interface as the reference lab: a module-level `documents` list, where
each document is a dict with id / title / is_current / text.

Here, documents are loaded from banking_knowledge_base_1000.csv instead of
being hardcoded, since there are 1,000 of them. Each CSV row becomes one
document: title = "Section: Question", text = Answer, is_current = True
(the source data has no archived/superseded notion, but the field is kept
so the CURRENT/OUTDATED logic in 06_retrieve_context.py still works
unchanged if you later add superseded policies).
"""

import csv
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banking_knowledge_base_1000.csv")


def _load_documents_from_csv(csv_path=CSV_PATH):
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            section = (row.get("Section") or "").strip()
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                continue
            docs.append({
                "id": f"doc_{i}",
                "title": f"{section}: {question}" if section else question,
                "is_current": True,
                "text": answer,
            })
    return docs


documents = _load_documents_from_csv()


if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents from '{CSV_PATH}'.")
    print("Example:", documents[0])
