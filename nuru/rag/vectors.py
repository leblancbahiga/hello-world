"""Utilitaires vectoriels : sérialisation compacte + top-k cosinus.

`numpy` est importé à la demande (lazy) et reste optionnel au runtime :
en son absence, une boucle Python suffit pour les corpus personnels.
"""

from __future__ import annotations

import array
import math
from typing import Any


def pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def unpack(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


def top_k_cosine(vectors: list[list[float]], query: list[float], k: int) -> list[tuple[int, float]]:
    """Retourne les indices des k vecteurs les plus proches de `query`.

    Le résultat est trié du meilleur au moins bon.
    """
    if not vectors:
        return []
    try:
        import numpy as np  # import différé

        matrix = np.asarray(vectors, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        scores = matrix @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]
    except ImportError:  # repli pur Python
        q = l2_normalize(query)
        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(vectors):
            v = l2_normalize(vec)
            dot = sum(a * b for a, b in zip(v, q))
            scored.append((dot, i))
        scored.sort(reverse=True)
        return [(i, score) for score, i in scored[:k]]
