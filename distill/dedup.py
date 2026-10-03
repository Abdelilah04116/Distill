"""Deduplication semantique.

Deux exemples peuvent etre differents mot pour mot et pourtant identiques pour
le modele. C'est le principal facteur d'uniformite d'un dataset synthetique, et
c'est aussi par la que fuite le test set quand une entree de test est reecrite
en legerement different.

Trois implementations, choisies dans cet ordre selon ce qui est installe :
  1. sentence-transformers + Qdrant (le chemin nominal du cahier des charges) ;
  2. sentence-transformers + numpy (si Qdrant n'est pas la) ;
  3. un encodeur de hachage de n-grammes en numpy pur, qui ne depend de rien.

Le troisieme est volontairement rustique : il attrape les quasi-doublons de
surface, ce qui suffit a faire tourner les tests et un petit run sans GPU.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import Example
from .logging_utils import get_logger

_WS = re.compile(r"\s+")
_HASH_DIM = 512
_NGRAM = 4


class Embedder(Protocol):
    """Tout ce dont la deduplication a besoin : des vecteurs normalises."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Encodeur sans dependance : hachage de n-grammes de caracteres.

    Moins fin qu'un vrai modele d'embedding, mais deterministe, instantane et
    suffisant pour reperer les reformulations superficielles.
    """

    name = "hashing-ngrams"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), _HASH_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            normalised = _WS.sub(" ", text.strip().lower())
            for i in range(max(1, len(normalised) - _NGRAM + 1)):
                gram = normalised[i : i + _NGRAM]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
                vectors[row, int.from_bytes(digest, "little") % _HASH_DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEmbedder:
    """Encodeur multilingue, le choix par defaut quand il est installe."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(model_name: str) -> Embedder:
    """Meilleur encodeur disponible, avec repli silencieux et trace."""
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - toute panne de chargement doit replier
        get_logger().warning(
            f"[yellow]sentence-transformers indisponible[/] ({type(exc).__name__}), "
            "repli sur l'encodeur de hachage (pip install 'distill-slm[dedup]' pour la version complete)"
        )
        return HashingEmbedder()


@dataclass
class DedupResult:
    kept: list[Example]
    removed: list[Example]
    leaked: list[Example]
    embedder: str
    threshold: float

    @property
    def stats(self) -> dict[str, object]:
        return {
            "kept": len(self.kept),
            "removed_duplicates": len(self.removed),
            "removed_test_leaks": len(self.leaked),
            "embedder": self.embedder,
            "threshold": self.threshold,
        }


def _greedy_filter(
    vectors: np.ndarray,
    threshold: float,
    protected: np.ndarray | None,
) -> tuple[list[int], list[int], list[int]]:
    """Garde un exemple s'il n'est proche d'aucun exemple deja garde.

    Les vecteurs sont normalises, donc le produit scalaire est le cosinus.
    """
    kept_idx: list[int] = []
    dup_idx: list[int] = []
    leak_idx: list[int] = []
    kept_matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)

    for i in range(vectors.shape[0]):
        vector = vectors[i]
        if (
            protected is not None
            and protected.shape[0]
            and float((protected @ vector).max()) > threshold
        ):
            leak_idx.append(i)
            continue
        if kept_matrix.shape[0] and float((kept_matrix @ vector).max()) > threshold:
            dup_idx.append(i)
            continue
        kept_idx.append(i)
        kept_matrix = np.vstack([kept_matrix, vector[None, :]])
    return kept_idx, dup_idx, leak_idx


def _qdrant_filter(
    vectors: np.ndarray,
    threshold: float,
    protected: np.ndarray | None,
) -> tuple[list[int], list[int], list[int]] | None:
    """Meme logique, mais l'index vectoriel fait la recherche du plus proche.

    Retourne None si Qdrant n'est pas installe, pour laisser la place au repli.
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        return None

    dim = int(vectors.shape[1])
    client = QdrantClient(":memory:")
    params = models.VectorParams(size=dim, distance=models.Distance.COSINE)
    client.create_collection("kept", vectors_config=params)

    protected_ready = protected is not None and protected.shape[0] > 0
    if protected_ready:
        client.create_collection("protected", vectors_config=params)
        assert protected is not None
        client.upsert(
            "protected",
            points=[
                models.PointStruct(id=i, vector=protected[i].tolist())
                for i in range(protected.shape[0])
            ],
        )

    kept_idx: list[int] = []
    dup_idx: list[int] = []
    leak_idx: list[int] = []
    for i in range(vectors.shape[0]):
        query = vectors[i].tolist()
        if protected_ready:
            near = client.query_points("protected", query=query, limit=1, with_payload=False).points
            if near and near[0].score > threshold:
                leak_idx.append(i)
                continue
        if kept_idx:
            near = client.query_points("kept", query=query, limit=1, with_payload=False).points
            if near and near[0].score > threshold:
                dup_idx.append(i)
                continue
        client.upsert("kept", points=[models.PointStruct(id=i, vector=query)])
        kept_idx.append(i)
    client.close()
    return kept_idx, dup_idx, leak_idx


def deduplicate(
    examples: Sequence[Example],
    *,
    threshold: float = 0.92,
    embedder: Embedder | None = None,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    protected: Sequence[Example] = (),
    use_qdrant: bool = True,
) -> DedupResult:
    """Deduplique un lot et ecarte ce qui ressemble trop au test set."""
    if not examples:
        return DedupResult([], [], [], embedder.name if embedder else "none", threshold)

    enc = embedder or get_embedder(embedding_model)
    vectors = enc.encode([ex.input for ex in examples])
    protected_vectors = enc.encode([ex.input for ex in protected]) if protected else None

    outcome = _qdrant_filter(vectors, threshold, protected_vectors) if use_qdrant else None
    if outcome is None:
        outcome = _greedy_filter(vectors, threshold, protected_vectors)
    kept_idx, dup_idx, leak_idx = outcome

    return DedupResult(
        kept=[examples[i] for i in kept_idx],
        removed=[examples[i] for i in dup_idx],
        leaked=[examples[i] for i in leak_idx],
        embedder=enc.name,
        threshold=threshold,
    )
