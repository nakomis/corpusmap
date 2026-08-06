#!/usr/bin/env python3
"""Build the demo corpus: eight public-domain books, split into passages.

Why eight books rather than one. A single novel clusters badly — it is one
author, one register, one subject, so the map comes out as a single smear and
teaches you nothing about what the tool does. A deliberately mixed shelf gives
clusters you can check by eye: if the recipes do not separate from the marine
biology, something is wrong with your pipeline and not with the world.

Why passages rather than sentences. A sentence is usually too short to have a
topic — "He said nothing." belongs to no cluster in any corpus — and embedding
models are trained on longer spans. Consecutive sentences are therefore glued
together up to `--max-chars`, never crossing a paragraph boundary. Roughly a
short paragraph each, which is about the smallest unit that reliably means
something on its own.

Everything here is out of copyright in the UK and the US, and Project
Gutenberg's own licence text is stripped, so the resulting dataset is public
domain and redistributable. See datasets/SOURCES.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request

import psycopg

# Chosen for topical distance from each other, not for literary merit. The
# point is that a working pipeline must put the recipes, the beetles and the
# submarines in different places.
BOOKS = [
    (1342, "Pride and Prejudice", "Jane Austen", "fiction-social"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "fiction-detective"),
    (164, "Twenty Thousand Leagues under the Sea", "Jules Verne", "fiction-adventure"),
    (2701, "Moby Dick", "Herman Melville", "fiction-adventure"),
    (1228, "On the Origin of Species", "Charles Darwin", "science-biology"),
    (10136, "The Voyage of the Beagle", "Charles Darwin", "science-travel"),
    (10681, "The Book of Household Management", "Isabella Beeton", "reference-cookery"),
    (2680, "Meditations", "Marcus Aurelius", "philosophy"),
]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Project Gutenberg wraps each text in a licence header and footer. They are
# long, formulaic and nearly identical between books — which means that left in
# place they form their own enormous, perfectly coherent cluster about the
# Project Gutenberg Literary Archive Foundation. Amusing once; useless twice.
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(]?[A-Z0-9])")

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    book_id     integer     NOT NULL,
    title       text        NOT NULL,
    author      text        NOT NULL,
    category    text        NOT NULL,
    seq         integer     NOT NULL,
    content     text        NOT NULL,
    embedding   vector({dims})
);

CREATE INDEX IF NOT EXISTS documents_category_idx ON documents (category);
-- Partial index on the NULL vectors: `corpusmap embed` polls for unembedded
-- rows in a loop, and without this that query is a sequential scan every time.
CREATE INDEX IF NOT EXISTS documents_unembedded_idx
    ON documents (id) WHERE embedding IS NULL;
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_book(book_id: int, cache_dir: str = ".cache") -> str:
    import pathlib

    cache = pathlib.Path(cache_dir)
    cache.mkdir(exist_ok=True)
    path = cache / f"pg{book_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")

    url = GUTENBERG_URL.format(id=book_id)
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "corpusmap/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    return text


def strip_licence(text: str) -> str:
    if m := START_RE.search(text):
        text = text[m.end():]
    if m := END_RE.search(text):
        text = text[: m.start()]
    return text.strip()


def paragraphs(text: str):
    """Blank-line separated blocks, with hard-wrapped lines rejoined.

    Gutenberg plain text is wrapped at ~70 characters, so a naive line-based
    split would cut sentences in half and produce fragments that embed as
    nonsense.
    """
    for block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in block.splitlines())
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            yield joined


def passages(text: str, max_chars: int, min_chars: int):
    """Consecutive sentences glued up to max_chars, never crossing a paragraph."""
    for para in paragraphs(text):
        # Chapter headings, roman numerals, all-caps front matter: too short to
        # have a topic, and they cluster on their own formatting.
        if len(para) < min_chars:
            continue
        buf = ""
        for sentence in SENTENCE_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if buf and len(buf) + 1 + len(sentence) > max_chars:
                yield buf
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if len(buf) >= min_chars:
            yield buf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", required=True)
    p.add_argument("--dims", type=int, default=768,
                   help="embedding dimensions; 768 for nomic-embed-text, "
                        "1024 for mxbai-embed-large")
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--per-book", type=int, default=2500,
                   help="cap passages per book, so one long book does not "
                        "dominate the map purely by being long")
    p.add_argument("--reset", action="store_true", help="drop and recreate the table")
    a = p.parse_args()

    with psycopg.connect(a.dsn) as conn:
        with conn.cursor() as cur:
            if a.reset:
                cur.execute("DROP TABLE IF EXISTS documents")
            cur.execute(SCHEMA.format(dims=a.dims))
        conn.commit()

        total = 0
        for book_id, title, author, category in BOOKS:
            try:
                raw = fetch_book(book_id)
            except Exception as e:  # noqa: BLE001
                log(f"!! skipping {title}: {e}")
                continue

            chunks = list(passages(strip_licence(raw), a.max_chars, a.min_chars))
            if len(chunks) > a.per_book:
                # Even stride rather than head-truncation: taking the first N
                # would give you the opening chapters of every book and none of
                # the endings, which is a different corpus than the one claimed.
                stride = len(chunks) / a.per_book
                chunks = [chunks[int(i * stride)] for i in range(a.per_book)]

            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents "
                    "(book_id, title, author, category, seq, content) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (book_id, title, author, category, i, c)
                        for i, c in enumerate(chunks)
                    ],
                )
            conn.commit()
            total += len(chunks)
            log(f"{title:<45} {len(chunks):>6,} passages")

    log(f"done: {total:,} passages across {len(BOOKS)} books")
    log("next: corpusmap embed --dsn ... --model nomic-embed-text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
