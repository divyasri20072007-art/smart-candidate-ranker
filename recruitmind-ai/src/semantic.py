"""
semantic.py
-----------
Provides semantic (meaning-based) similarity between the JD and each
candidate's profile text — the core piece that lets RecruitMind AI go
beyond literal keyword matching.

Two backends:
  1. SentenceTransformer embeddings (best quality) — used automatically
     if the `sentence-transformers` package AND a local/cached model are
     available.
  2. TF-IDF + cosine similarity (always available, zero internet
     dependency) — a strong, fast, fully-offline fallback that still
     captures shared vocabulary/context rather than exact string matches
     because it is fit jointly over the whole corpus (JD + all
     candidates), so synonyms used across many candidate profiles end up
     in similar regions of the vector space.

The rest of the pipeline only calls `SemanticIndex.similarity(text_a,
text_b)`, so swapping in a stronger embedding model later (OpenAI,
Cohere, a fine-tuned bi-encoder, etc.) only requires changing this file.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class _TfidfBackend:
    name = "tfidf"

    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def encode(self, text):
        return self.vectorizer.transform([text])

    def similarity(self, text_a, text_b):
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        return float(cosine_similarity(vec_a, vec_b)[0][0])


class _SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._cache = {}

    def _embed(self, text):
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def similarity(self, text_a, text_b):
        import numpy as np

        a = self._embed(text_a)
        b = self._embed(text_b)
        return float(np.dot(a, b))


class SemanticIndex:
    """Fits once over the full corpus, then answers pairwise similarity
    queries cheaply for the rest of the run."""

    def __init__(self, corpus, prefer_embeddings=True):
        self.backend = None
        if prefer_embeddings:
            try:
                self.backend = _SentenceTransformerBackend()
            except Exception:
                self.backend = None
        if self.backend is None:
            self.backend = _TfidfBackend(corpus)

    @property
    def backend_name(self):
        return self.backend.name

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        score = self.backend.similarity(text_a, text_b)
        # Clip into [0, 1] — cosine sim on TF-IDF is already >=0,
        # but embedding cosine similarity can dip slightly negative.
        return max(0.0, min(1.0, score))
