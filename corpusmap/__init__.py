"""corpusmap — cluster and label a text dataset.

The pipeline is four stages, each usable on its own:

    embed    text            -> vectors        (Ollama, into pgvector)
    project  vectors         -> 3D coordinates (normalise -> PCA -> UMAP)
    cluster  3D coordinates  -> cluster ids    (HDBSCAN)
    label    cluster ids     -> English names  (c-TF-IDF, optionally an LLM)

Nothing here is specific to any one corpus. Point it at a Postgres table with
a text column and a pgvector column and it will do the rest.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
