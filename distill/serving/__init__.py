"""Service HTTP du modele distille."""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str) -> object:
    """Import tardif : FastAPI n'est requis que pour `distill serve`."""
    if name == "app":
        from .api import app

        return app
    raise AttributeError(name)
