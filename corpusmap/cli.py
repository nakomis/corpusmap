"""Command line entry point.

Four subcommands mirroring the four stages, plus ``run`` which does the lot:

    corpusmap embed    --dsn ... --model nomic-embed-text
    corpusmap project  --dsn ... --out build/
    corpusmap label    --dsn ... --out build/ --llm llama3.2:3b
    corpusmap run      --dsn ... --out build/ --llm llama3.2:3b

``project`` and ``label`` are separate on purpose. Labelling is the stage that
takes hours, and you will want to re-run it — with a different stop list, a
different model, or after the model rejected half your labels — without
re-fitting anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .db import Corpus, fetch, fetch_text
from .export import projection_doc, write
from .label import LlmConfig, ctfidf, gather, label_clusters, load_stop_words
from .project import ProjectionConfig, facets, project


def _corpus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dsn", required=True, help="Postgres connection string")
    p.add_argument("--table", default="documents")
    p.add_argument("--id-col", default="id")
    p.add_argument("--text-col", default="content")
    p.add_argument("--vector-col", default="embedding")
    p.add_argument(
        "--facet-col",
        default=None,
        help="column to fit a separate projection per value of "
        "(author, source, language...). Each value gets its own fit, "
        "because UMAP projections do not compose.",
    )
    p.add_argument("--where", default=None, help="extra SQL predicate")
    p.add_argument(
        "--meta-cols",
        default=None,
        help="comma-separated display-only columns shown on the viewer's hover "
        "card (e.g. title,author,chapter). Never used by the mathematics.",
    )


def _meta_cols(a: argparse.Namespace) -> tuple[str, ...]:
    raw = getattr(a, "meta_cols", None)
    return tuple(c.strip() for c in raw.split(",") if c.strip()) if raw else ()


def _corpus(a: argparse.Namespace) -> Corpus:
    return Corpus(
        dsn=a.dsn,
        table=a.table,
        id_col=a.id_col,
        text_col=a.text_col,
        vector_col=a.vector_col,
        facet_col=getattr(a, "facet_col", None),
        where=a.where,
        meta_cols=_meta_cols(a),
    )


def _projection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-dims", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--min-cluster-size", type=int, default=20)
    p.add_argument(
        "--max-cluster-size",
        type=int,
        default=2000,
        help="cap on cluster size, forcing HDBSCAN further down the condensed "
        "tree rather than accepting one featureless blob. 0 disables.",
    )
    p.add_argument("--out", default="build", help="output directory")
    p.add_argument("--prefix", default="map")
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="skip the self-contained HTML map and write only the JSON artefacts",
    )
    p.add_argument(
        "--corpus-name",
        default=None,
        help="name shown in the viewer's panel (default: the table name)",
    )
    p.add_argument(
        "--snippet-chars",
        type=int,
        default=320,
        help="how much of each document the viewer's hover card shows. This is "
        "the main lever on the size of the HTML artefact.",
    )


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--llm",
        metavar="MODEL",
        default=None,
        help="refine labels into English with this Ollama model, "
        "e.g. llama3.2:3b. Omitted, labels are the raw c-TF-IDF terms.",
    )
    p.add_argument("--llm-endpoint", default="http://127.0.0.1:11434")
    p.add_argument("--llm-workers", type=int, default=4)
    p.add_argument(
        "--llm-min-cluster-size",
        type=int,
        default=0,
        help="skip the model for clusters smaller than this and use the terms. "
        "Labelling is ~98%% of runtime and half of all clusters sit near the "
        "floor, so this is the cheapest speed-up available.",
    )
    p.add_argument("--stop-words", default=None, help="extra stop words, one per line")
    p.add_argument("--top-n", type=int, default=12)


def _cfg(a: argparse.Namespace) -> ProjectionConfig:
    return ProjectionConfig(
        seed=a.seed,
        pca_dims=a.pca_dims,
        n_neighbors=a.n_neighbors,
        min_dist=a.min_dist,
        metric=a.metric,
        min_cluster_size=a.min_cluster_size,
        max_cluster_size=a.max_cluster_size or None,
    )


def _llm_cfg(a: argparse.Namespace) -> LlmConfig | None:
    if not a.llm:
        return None
    return LlmConfig(
        model=a.llm,
        endpoint=a.llm_endpoint,
        workers=a.llm_workers,
        min_cluster_size=a.llm_min_cluster_size,
    )


def cmd_embed(a: argparse.Namespace) -> int:
    from .embed import embed_corpus

    n = embed_corpus(
        _corpus(a),
        model=a.model,
        endpoint=a.llm_endpoint,
        truncate=a.truncate,
        batch=a.batch,
        workers=a.workers,
    )
    print(f"embedded {n:,} documents")
    return 0


def _do_project(a: argparse.Namespace) -> list:
    corpus = _corpus(a)
    ids, X, facet_values = fetch(corpus)
    cfg = _cfg(a)

    if corpus.facet_col:
        splits = facets(ids, X, facet_values)
    else:
        import numpy as np

        splits = [("all", np.ones(len(ids), dtype=bool))]

    projections = []
    for name, mask in splits:
        sub_ids = [ids[i] for i in mask.nonzero()[0]]
        if len(sub_ids) < a.min_cluster_size:
            print(f"skipping {name}: only {len(sub_ids)} documents")
            continue
        projections.append(project(sub_ids, X[mask], cfg, name=name))
    return projections


def _do_label(a: argparse.Namespace, projections: list) -> list[dict]:
    corpus = _corpus(a)
    llm = _llm_cfg(a)
    stop = load_stop_words(a.stop_words)
    text_by_id = fetch_text(corpus)

    # One cache across every facet. Facets overlap heavily — `all` contains
    # every point in all the others — so the same topic is named repeatedly
    # within a single run, and keying on the term tuple makes the repeats free.
    cache: dict = {}
    docs = []
    for proj in projections:
        pseudo, samples = gather(proj.ids, proj.labels, text_by_id, stop)
        terms = ctfidf(pseudo, a.top_n)
        labels = label_clusters(
            terms, samples, proj.cluster_sizes(), llm=llm, cache=cache
        )
        docs.append(projection_doc(proj, labels))
    return docs


def _emit(a: argparse.Namespace, docs: list[dict]) -> None:
    """Write the JSON artefacts and, unless told not to, the viewer."""
    path = write(a.out, a.prefix, docs, extra={"corpusmap": __version__})
    print(f"wrote {path}")
    if getattr(a, "no_viewer", False):
        return

    from . import pack
    from .db import fetch_meta
    from .export import write_docs

    # The hover-card table is written as its own artefact rather than folded
    # into the projections, so that `corpusmap view` can rebuild the HTML later
    # without a database to hand.
    corpus = _corpus(a)
    write_docs(
        a.out,
        a.prefix,
        docs,
        text_by_id=fetch_text(corpus),
        meta_by_id=fetch_meta(corpus),
        meta_names=list(corpus.meta_cols),
        snippet_chars=a.snippet_chars,
    )
    html = pack.build_from_dir(
        a.out, a.prefix, corpus=getattr(a, "corpus_name", None) or a.table
    )
    print(f"wrote {html}  <- open this")


def cmd_project(a: argparse.Namespace) -> int:
    projections = _do_project(a)
    docs = [projection_doc(p) for p in projections]
    # Coordinates are cached so `label` can run without re-fitting.
    _save_state(a, projections)
    _emit(a, docs)
    return 0


def cmd_label(a: argparse.Namespace) -> int:
    projections = _load_state(a)
    if not projections:
        sys.exit(f"no cached projections under {a.out}; run `corpusmap project` first")
    _emit(a, _do_label(a, projections))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    projections = _do_project(a)
    _save_state(a, projections)
    _emit(a, _do_label(a, projections))
    return 0


def cmd_view(a: argparse.Namespace) -> int:
    """Rebuild the viewer from artefacts a previous run already wrote."""
    from . import pack

    html = pack.build_from_dir(a.out, a.prefix, out_path=a.html, corpus=a.corpus)
    print(f"wrote {html}")
    return 0


def _state_path(a: argparse.Namespace) -> Path:
    return Path(a.out) / f"{a.prefix}-state.npz"


def _save_state(a: argparse.Namespace, projections: list) -> None:
    """Cache the fit so labelling can be re-run without re-projecting."""
    import numpy as np

    Path(a.out).mkdir(parents=True, exist_ok=True)
    payload = {}
    meta = []
    for i, p in enumerate(projections):
        payload[f"coords_{i}"] = p.coords
        payload[f"labels_{i}"] = p.labels
        payload[f"ids_{i}"] = np.asarray(p.ids, dtype=object)
        meta.append({"name": p.name, "seed": p.seed})
    payload["meta"] = np.asarray([json.dumps(meta)], dtype=object)
    np.savez_compressed(_state_path(a), **payload)


def _load_state(a: argparse.Namespace) -> list:
    import numpy as np

    from .project import Projection

    path = _state_path(a)
    if not path.exists():
        return []
    z = np.load(path, allow_pickle=True)
    meta = json.loads(z["meta"][0])
    cfg = _cfg(a)
    return [
        Projection(
            name=m["name"],
            ids=list(z[f"ids_{i}"]),
            coords=z[f"coords_{i}"],
            labels=z[f"labels_{i}"],
            seed=m["seed"],
            config=cfg,
        )
        for i, m in enumerate(meta)
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corpusmap",
        description="Cluster and label a text dataset held in Postgres/pgvector.",
    )
    p.add_argument("--version", action="version", version=f"corpusmap {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="fill the vector column via Ollama")
    _corpus_args(e)
    e.add_argument("--model", default="nomic-embed-text")
    e.add_argument("--llm-endpoint", default="http://127.0.0.1:11434")
    e.add_argument("--truncate", type=int, default=2000)
    e.add_argument("--batch", type=int, default=200)
    e.add_argument("--workers", type=int, default=4)
    e.set_defaults(func=cmd_embed)

    pr = sub.add_parser("project", help="reduce to 3D and cluster")
    _corpus_args(pr)
    _projection_args(pr)
    pr.set_defaults(func=cmd_project)

    la = sub.add_parser("label", help="name the clusters (re-uses a cached fit)")
    _corpus_args(la)
    _projection_args(la)
    _llm_args(la)
    la.set_defaults(func=cmd_label)

    ru = sub.add_parser("run", help="project and label in one go")
    _corpus_args(ru)
    _projection_args(ru)
    _llm_args(ru)
    ru.set_defaults(func=cmd_run)

    vi = sub.add_parser(
        "view", help="rebuild the self-contained HTML map from existing artefacts"
    )
    vi.add_argument("--out", default="build", help="directory holding the artefacts")
    vi.add_argument("--prefix", default="map")
    vi.add_argument("--html", default=None, help="output path (default <out>/<prefix>.html)")
    vi.add_argument("--corpus", default=None, help="name shown in the panel")
    vi.set_defaults(func=cmd_view)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
