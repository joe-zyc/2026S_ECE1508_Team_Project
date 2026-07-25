"""SentenceTransformer embedding provider."""

import threading

import numpy as np

from .exceptions import EmbeddingUnavailableError


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, expected_dimension: int) -> None:
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    "Embedding model could not be loaded"
                ) from exc

    def embed(self, text: str) -> list[float]:
        self.load()
        try:
            vector = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            values = np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                "Embedding model could not encode the query"
            ) from exc
        if values.shape != (self.expected_dimension,):
            raise EmbeddingUnavailableError(
                f"Embedding model returned dimension {values.shape}; "
                f"expected ({self.expected_dimension},)"
            )
        return values.tolist()
