"""Cache disque des reponses du professeur.

Deux raisons d'exister, toutes deux dans le cahier des charges : plafonner le
cout de generation, et rendre un run reprenable apres un crash sans repayer ce
qui a deja ete produit. La cle couvre tout ce qui peut changer la reponse, donc
un changement de modele ou de prompt invalide naturellement l'entree.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    """Empreinte stable d'un appel (tri des cles pour que l'ordre ne compte pas)."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """Cache cle-valeur sur disque, un fichier JSON par entree.

    Volontairement simple : pas de base, pas de verrou inter-processus. Les
    ecritures sont atomiques, donc deux runs concurrents se marchent dessus
    sans corrompre le cache.
    """

    def __init__(self, directory: Path | str, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Deux niveaux de prefixe : evite des dizaines de milliers de fichiers
        # a plat dans un seul dossier.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        with self._lock:
            self.hits += 1
        return payload if isinstance(payload, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 3)}

    def clear(self) -> int:
        """Vide le cache, retourne le nombre d'entrees supprimees."""
        if not self.directory.is_dir():
            return 0
        removed = 0
        for file in self.directory.rglob("*.json"):
            file.unlink(missing_ok=True)
            removed += 1
        return removed
