"""Domain exceptions mapped to stable HTTP responses."""


class BackendError(Exception):
    """Base class for expected backend failures."""


class UpstreamModelError(BackendError):
    """The query or recommendation language model failed."""


class DatabaseUnavailableError(BackendError):
    """The product database could not serve the request."""


class EmbeddingUnavailableError(BackendError):
    """The embedding model could not serve the request."""


class InvalidModelOutputError(UpstreamModelError):
    """A model returned output that violated the requested contract."""
