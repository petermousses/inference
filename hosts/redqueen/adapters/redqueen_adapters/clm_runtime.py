"""Run pinned CLM with bounded connects and unbounded encoder inference reads."""
from __future__ import annotations

from importlib.metadata import version


def configure_embedder(module) -> None:
    """Keep CLM's vector/cache/math implementation and health probe unchanged.

    contrastive-lm 0.1.0 does not expose its encoder timeout in the server CLI.
    Its Embedder passes this tuple directly to requests.Session.post.
    """
    original = module.Embedder

    class UnboundedEmbedder(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout = (10.0, None)

    module.Embedder = UnboundedEmbedder


def main() -> None:
    if version("contrastive-lm") != "0.1.0":
        raise SystemExit("CLM runtime pin changed; validate the encoder timeout integration")
    import clm.embedder
    configure_embedder(clm.embedder)
    from clm.server import main as serve
    serve()


if __name__ == "__main__":
    main()
