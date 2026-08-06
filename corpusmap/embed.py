"""Turn text into vectors with Ollama, and write them back to Postgres.

This stage is optional. If your table already has a populated pgvector column —
because something else in your system embeds as it ingests — skip straight to
``corpusmap project``.

An embedding is the document's "meaning fingerprint": a fixed-length list of
numbers where documents about similar things end up with similar numbers. Every
later stage in this tool operates on the fingerprints and never looks at the
text again until it is time to write labels.

Two properties of embedding models bite people, so they are worth stating:

- **Context is short.** ``nomic-embed-text`` handles 8,192 tokens but
  ``mxbai-embed-large`` handles 512. Anything past the limit is silently
  dropped, so a long document gets embedded on its opening paragraph and you
  never find out. We truncate explicitly, at a length you choose, so at least
  the truncation is yours.
- **Dimensions are not comparable.** A 768-dimension nomic vector and a
  1024-dimension mxbai vector cannot be mixed in one column, and re-embedding a
  corpus with a different model means re-running everything downstream.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import psycopg

from .db import Corpus


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def embed_one(
    text: str,
    model: str,
    endpoint: str = "http://127.0.0.1:11434",
    timeout: int = 120,
) -> list[float] | None:
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["embedding"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        _log(f"  embed failed ({e})")
        return None


def embed_corpus(
    corpus: Corpus,
    model: str,
    endpoint: str = "http://127.0.0.1:11434",
    truncate: int = 2000,
    batch: int = 200,
    workers: int = 4,
) -> int:
    """Embed every row whose vector column is still NULL. Resumable.

    Resumability is not a nicety. Embedding a corpus of any size takes long
    enough that something will interrupt it — a laptop lid, a dropped LAN, an
    Ollama restart — and starting again from zero each time is how people give
    up on this. Because the query selects on ``IS NULL`` and each batch commits,
    re-running simply picks up where it stopped.
    """
    done = 0
    t0 = time.perf_counter()

    with psycopg.connect(corpus.dsn) as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {corpus.id_col}::text, {corpus.text_col} "
                    f"FROM {corpus.table} "
                    f"WHERE {corpus.vector_col} IS NULL "
                    f"  AND {corpus.text_col} IS NOT NULL "
                    f"LIMIT {batch}"
                )
                rows = cur.fetchall()
            if not rows:
                break

            def work(row):
                doc_id, text = row
                return doc_id, embed_one(text[:truncate], model, endpoint)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(work, rows))

            with conn.cursor() as cur:
                for doc_id, vec in results:
                    if vec is None:
                        continue
                    cur.execute(
                        f"UPDATE {corpus.table} SET {corpus.vector_col} = %s "
                        f"WHERE {corpus.id_col}::text = %s",
                        (json.dumps(vec), doc_id),
                    )
            conn.commit()

            done += sum(1 for _, v in results if v is not None)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            _log(f"embedded {done:,} ({rate:.1f}/s)")

    return done
