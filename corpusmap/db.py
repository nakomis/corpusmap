"""Reading a corpus out of Postgres.

The whole tool is built around one assumption: your data is a table with an id,
some text, and a pgvector column. Everything else — what the table is called,
what the columns are called, whether there is a category column to facet on —
is configuration rather than code.

The one thing worth knowing before you read this file is that fetching
embeddings out of pgvector is usually the slowest stage of the whole pipeline
that is not the LLM. pgvector serialises vectors as *text* on the wire —
``[0.0123,-0.4567,...]`` — so a 768-dimension vector arrives as roughly 10 KB
of ASCII that then has to be parsed. On a 76,000-row corpus over a LAN that
measured 89 seconds, against 520 seconds for all of the actual mathematics.

``pgvector.psycopg`` fixes this if it is installed: it registers a binary
adaptor and the vectors arrive as numpy arrays with no parsing at all. We use
it when present and fall back to text parsing when it is not, because the
fallback works everywhere and the dependency is not worth making mandatory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import psycopg


@dataclass(frozen=True)
class Corpus:
    """Where the text and the vectors live.

    ``facet_col`` is optional and is the one piece of this that people
    consistently want and do not expect to get. It names a column to split the
    corpus on — author, source, language, year — so that you can fit a separate
    projection per value. See ``facets()`` for why separate fits are necessary
    rather than merely tidy.
    """

    dsn: str
    table: str = "documents"
    id_col: str = "id"
    text_col: str = "content"
    vector_col: str = "embedding"
    facet_col: str | None = None
    where: str | None = None
    #: Extra columns carried through to the viewer's hover card — title,
    #: author, url, timestamp, whatever identifies a document to a human.
    #: They are never used by the mathematics, only displayed.
    meta_cols: tuple[str, ...] = ()

    def _predicate(self) -> str:
        clauses = [f"{self.vector_col} IS NOT NULL"]
        if self.where:
            clauses.append(f"({self.where})")
        return " AND ".join(clauses)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(corpus: Corpus) -> tuple[list[str], np.ndarray, list[str | None]]:
    """Read every embedded row. Returns (ids, vectors, facet values).

    Read once, subset in memory. It is tempting to issue one query per facet,
    but the fetch is the expensive part and the filtering is free, so a
    five-facet run done that way costs five times the wait for no benefit.
    """
    facet_expr = f", {corpus.facet_col}::text" if corpus.facet_col else ""
    sql = (
        f"SELECT {corpus.id_col}::text, {corpus.vector_col}{facet_expr} "
        f"FROM {corpus.table} WHERE {corpus._predicate()}"
    )

    t0 = time.perf_counter()
    with psycopg.connect(corpus.dsn) as conn:
        _register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    if not rows:
        raise SystemExit(
            f"no rows in {corpus.table} with a non-null {corpus.vector_col}. "
            f"Have you run `corpusmap embed` yet?"
        )

    ids = [r[0] for r in rows]
    vectors = np.asarray([_as_vector(r[1]) for r in rows], dtype=np.float32)
    facets = [r[2] for r in rows] if corpus.facet_col else [None] * len(rows)

    _log(
        f"fetched {len(ids):,} x {vectors.shape[1]} vectors "
        f"in {time.perf_counter() - t0:.0f}s"
    )
    return ids, vectors, facets


def fetch_text(corpus: Corpus) -> dict[str, str]:
    """Read the text bodies, keyed by id. Needed only for labelling.

    Kept separate from ``fetch`` on purpose: projecting does not need the text
    at all, and on a large corpus the text is the bulk of the bytes.
    """
    sql = (
        f"SELECT {corpus.id_col}::text, {corpus.text_col} "
        f"FROM {corpus.table} WHERE {corpus._predicate()}"
    )
    t0 = time.perf_counter()
    with psycopg.connect(corpus.dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        out = {row[0]: row[1] for row in cur}
    _log(f"fetched {len(out):,} text bodies in {time.perf_counter() - t0:.0f}s")
    return out


def fetch_meta(corpus: Corpus) -> dict[str, list]:
    """Read the display-only columns, keyed by id.

    Returns an empty mapping when no ``meta_cols`` are configured, so callers
    never have to special-case the common setup.
    """
    if not corpus.meta_cols:
        return {}
    cols = ", ".join(corpus.meta_cols)
    sql = (
        f"SELECT {corpus.id_col}::text, {cols} "
        f"FROM {corpus.table} WHERE {corpus._predicate()}"
    )
    with psycopg.connect(corpus.dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return {row[0]: list(row[1:]) for row in cur}


def _register_vector(conn: psycopg.Connection) -> None:
    """Use pgvector's binary adaptor when it is installed."""
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except Exception:  # noqa: BLE001 — any failure just means the text path
        pass


def _as_vector(value) -> np.ndarray:
    """Cope with every shape a pgvector column arrives in.

    There are three, which is two more than you would hope:

    - ``numpy.ndarray`` — newer ``pgvector.psycopg`` with the binary adaptor.
    - ``pgvector.Vector`` — older ones return their own wrapper class, which is
      *not* an ndarray and does not subclass one. It exposes ``to_numpy()``.
    - ``str`` — no adaptor registered, so the raw text wire format arrives:
      ``'[0.1,-0.2,...]'``, which happens to be valid JSON.
    """
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(json.loads(value), dtype=np.float32)
