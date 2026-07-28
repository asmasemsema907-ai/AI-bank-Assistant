from importlib import import_module
import logging
import os

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

preprocessing = import_module("02_preprocessing")
chunks = import_module("03_chunking").build_chunks()

ALPHA = 0.6
# Swapped from the English-only "all-MiniLM-L6-v2" to a true multilingual
# model: this one was trained so that a sentence and its translation land
# close together in vector space across ~50 languages including Arabic,
# which is what actually makes semantic retrieval work for Arabic questions
# against this English-only knowledge base (previously only the small
# ARABIC_TERM_BRIDGE dictionary below could bridge the gap, and only for
# terms explicitly listed in it). Same 384-dim output as before, so nothing
# else in the pipeline needs to change.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

logger = logging.getLogger(__name__)

# NOTE ON BILINGUAL SUPPORT: the embedding model above is now multilingual
# (paraphrase-multilingual-MiniLM-L12-v2), so Arabic questions get real
# semantic retrieval against the English knowledge base, not just exact
# keyword matches. This small bridge dictionary is kept as an extra lexical
# boost for BM25 specifically (BM25 itself is still purely literal-token
# matching, so it still benefits from seeing the English term explicitly) —
# it's a helpful addition now, not the only thing making Arabic work.
ARABIC_TERM_BRIDGE = {
    "حساب التوفير": "savings account", "الحساب الجاري": "current account",
    "وديعة ثابتة": "fixed deposit", "اعرف عميلك": "kyc know your customer",
    "كلمة المرور": "password reset", "بطاقة الخصم": "debit card",
    "بطاقة الائتمان": "credit card", "قرض": "loan",
    "الفائدة": "interest rate", "التصنيف الائتماني": "credit score",
    "تحويل الأموال": "transfer money neft rtgs", "رصيد": "balance",
    "غسل الأموال": "money laundering aml", "صراف آلي": "atm",
    "الخدمات المصرفية عبر الهاتف": "mobile banking", "شيك": "cheque",
    "حساب مشترك": "joint account", "قسط شهري": "emi installment",
}


def _bridge_arabic_terms(text: str) -> str:
    """Appends English equivalents for any recognized Arabic banking terms."""
    additions = [en for ar, en in ARABIC_TERM_BRIDGE.items() if ar in text]
    return f"{text} {' '.join(additions)}".strip() if additions else text

tokenized_chunks = [chunk["search_text"].split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

SEARCH_TEXTS = [chunk["search_text"] for chunk in chunks]


class TfidfEmbeddingModel:
    """Local embedding fallback used when the transformer model is not cached."""

    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.vectorizer.fit(texts)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        matrix = self.vectorizer.transform(texts)
        if normalize_embeddings:
            matrix = normalize(matrix)
        if convert_to_numpy:
            return matrix.toarray().astype(np.float32)
        return matrix


def _load_embedding_model():
    try:
        return (
            SentenceTransformer(MODEL_NAME, local_files_only=not ALLOW_MODEL_DOWNLOAD),
            "sentence_transformer_multilingual",
        )
    except Exception:
        logger.warning(
            "Falling back to local TF-IDF embeddings because %s is not available locally.",
            MODEL_NAME,
        )
        return TfidfEmbeddingModel(SEARCH_TEXTS), "tfidf_local"


model, EMBEDDING_BACKEND_ID = _load_embedding_model()
chunk_embeddings = model.encode(
    SEARCH_TEXTS,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
EMBEDDING_DIMENSION = int(chunk_embeddings.shape[1]) if chunk_embeddings.size else 0


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(query, k=4):
    if not query or not query.strip() or k <= 0:
        return []

    clean_query = preprocessing.preprocess_text(_bridge_arabic_terms(query))
    if not clean_query:
        return []

    bm25_scores = bm25.get_scores(clean_query.split())
    query_embedding = model.encode(
        [clean_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embedding_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()

    hybrid_scores = ((1 - ALPHA) * min_max_normalize(bm25_scores)) + (
        ALPHA * min_max_normalize(embedding_scores)
    )

    ranking = np.argsort(hybrid_scores)[::-1][: min(k, len(chunks))]
    return [
        {**chunks[index], "score": hybrid_scores[index]}
        for index in ranking
    ]