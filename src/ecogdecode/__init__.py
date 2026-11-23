"""Causal decoding of individual finger flexion from human ECoG (BCI Competition IV, data set 4)."""

__version__ = "0.1.0"
__all__ = ["OnlineDecoder"]


def __getattr__(name: str):
    if name == "OnlineDecoder":  # imported lazily: "import ecogdecode" stays light
        from .online import OnlineDecoder

        return OnlineDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
