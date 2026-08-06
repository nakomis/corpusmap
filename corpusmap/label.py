"""Give every cluster a name.

A coloured blob labelled "cluster 223" tells a reader nothing. The map only
becomes interesting once you can see that *this* lobe is naval navigation and
*that* one is recipes involving suet.

Labels come from **class-based TF-IDF**, not from an LLM. The distinction
matters and it is the opposite of what most people reach for:

- c-TF-IDF asks "which terms are frequent in this cluster and rare in the rest
  of the corpus", which is *exactly* the question a label answers. It is
  deterministic, free, and runs over the whole corpus in seconds.
- An LLM asked to name a cluster has to be shown a sample, and a sample of a
  3,000-document cluster is a guess about the other 2,990.

So the LLM, when enabled, is a **second pass that turns terms into readable
English**. It renames; it does not decide. With it switched off you get the raw
terms, which are uglier but never wrong about what the cluster contains.

Ordinary TF-IDF would not work here. It weights terms per *document*, and a
document is one passage, so the top terms would be whatever that passage
happened to repeat. Concatenating each cluster into a single pseudo-document
first is what makes the comparison between clusters rather than between
documents.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# A general-purpose English stop list. Deliberately NOT extended with
# domain vocabulary — see `load_stop_words` for why that has to be your job.
STOP_WORDS = set(
    """
a about after all also am an and any are as at be because been before being but
by can cant could did do does doing done dont down each else even ever every few
for from further get got had has have having he her here hers him his how i if in
into is it its itself just like me more most much must my no nor not now of off
on once only or other our out over own re same she should so some such than that
the their them then there these they this those through to too under until up us
use used using very was we well were what when where which while who whom why
will with would you your yours
ok okay yes yeah yep nope thanks thank please sure right sorry
one two three new make made need needs want added adding change changed
think know see look looking try trying work working thing things way ways
let lets going go went back time first last still bit really actually
""".split()
)

WORD = re.compile(r"[a-z][a-z0-9_.-]{2,}")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_stop_words(path: str | None) -> set[str]:
    """Extend the base stop list from a file, one word per line.

    You will need this, and it is worth understanding why it cannot be
    shipped. Every corpus has content words that are perfectly good English and
    yet appear in nearly every cluster, so they carry no discriminating power
    and dominate every label. In a corpus of software chat logs those were
    "code", "file", "error", "command". In a corpus of Victorian novels they
    are "said", "man", "lady", "house".

    There is no general list, because the words that mean nothing *here* are
    exactly the words that mean everything somewhere else. Run the pipeline
    once without a list, read the labels, and put the words that turned up
    everywhere into a file.
    """
    if not path:
        return set(STOP_WORDS)
    with open(path, encoding="utf-8") as f:
        extra = {w.strip().lower() for w in f if w.strip() and not w.startswith("#")}
    _log(f"stop words: {len(STOP_WORDS)} base + {len(extra)} from {path}")
    return set(STOP_WORDS) | extra


def is_junk(word: str) -> bool:
    """Machine-generated identifiers, which are never topics.

    Temp paths, hashes and UUID fragments cluster beautifully — they really do
    co-occur — so c-TF-IDF scores them highly and they crowd out the word that
    would have explained the cluster. Two cheap signals catch nearly all of
    them: implausible length, and letters and digits interleaved rather than a
    word with a version number stuck on the end.
    """
    if len(word) > 18:
        return True
    digits = sum(c.isdigit() for c in word)
    return 3 <= digits < len(word) and not word[0].isdigit()


def tokens(text: str, stop: set[str]):
    for w in WORD.findall(text.lower()):
        w = w.strip("._-")
        if len(w) > 2 and w not in stop and not w.isdigit() and not is_junk(w):
            yield w


def ctfidf(docs: dict[int, Counter], top_n: int = 12) -> dict[int, list[str]]:
    """Top terms per cluster by class-based TF-IDF.

    ``docs`` maps cluster id -> Counter of term frequencies for the *whole*
    cluster. The weight is the term's frequency within its cluster
    (L1-normalised) multiplied by ``log(1 + n_clusters / clusters_containing)``
    — which rewards a term for being common **here** and absent **elsewhere**.
    """
    n_clusters = len(docs)
    doc_freq: Counter = Counter()
    for counts in docs.values():
        doc_freq.update(counts.keys())

    out: dict[int, list[str]] = {}
    for cid, counts in docs.items():
        total = sum(counts.values()) or 1
        scored = [
            (t, (f / total) * math.log(1 + n_clusters / doc_freq[t]))
            for t, f in counts.items()
            # A term appearing exactly once in a large cluster is noise, not a
            # topic. In a small cluster it may be all the evidence there is.
            if f >= 2 or total < 200
        ]
        scored.sort(key=lambda x: -x[1])
        out[cid] = [t for t, _ in scored[:top_n]]
    return out


def terms_label(terms: list[str], n: int = 3) -> str | None:
    """Fallback label: the top few terms, comma-joined."""
    return ", ".join(terms[:n]) if terms else None


def grounded(label: str | None, terms: list[str], n: int = 6) -> bool:
    """Does the label actually mention something the cluster is about?

    This function is the single most valuable thing in this file, so it gets
    the longest comment.

    A small model asked to name a cluster it does not understand **does not say
    so**. It emits a confident generic phrase. Measured over 1,460 clusters
    with llama3.2:3b, the failure was completely consistent and always in the
    same direction:

        ddr3, sata, fpga     -> "Software Engineering Chat Log"
        great, nah, mornin   -> "Software engineering meeting discussion"
        spacex, spcx, hedge  -> "Software Engineering Chat Log"

    None of those is a paraphrase. They are inventions, and each is *worse*
    than the terms it replaced, because it reads as authoritative whilst saying
    nothing. The cluster about SpaceX and hedging is not a software engineering
    chat log.

    So a label is only kept if it is anchored in the evidence: at least one of
    the top terms must appear in it.

    The matching rule is deliberately narrow, and got narrower after a bug:

    - A term longer than five characters matches on its **first four
      characters at a word boundary**, so "testing" grounds "tests" and
      "cognito" grounds "Cognito". Inflections and capitalisation are exactly
      what you want to forgive.
    - A term of five characters or fewer must match as a **whole word**. A
      four-character prefix of a short word matches far too much, and a plain
      substring test is worse still: it let "sata" ground "Satanic verses",
      which is precisely the kind of accidental agreement this function exists
      to refuse.

    This costs nothing, needs no second model, and turns an unusable labelling
    pass into a usable one.
    """
    if not label or not terms:
        return False
    low = label.lower()
    for t in terms[:n]:
        if len(t) > 5:
            pattern = r"\b" + re.escape(t[:4])
        else:
            pattern = r"\b" + re.escape(t) + r"\b"
        if re.search(pattern, low):
            return True
    return False


@dataclass
class LlmConfig:
    """Settings for the optional English-prettifying pass."""

    model: str = "llama3.2:3b"
    endpoint: str = "http://127.0.0.1:11434"
    timeout: int = 120
    workers: int = 4
    min_cluster_size: int = 0  # skip clusters smaller than this; 0 = label all


def llm_label(terms: list[str], samples: list[str], cfg: LlmConfig) -> str | None:
    """Turn terms into a short English phrase. Returns None on any failure.

    Deliberately best-effort. A label is decoration, and a run that aborted
    because a local model was not loaded would be a far worse outcome than one
    that fell back to the raw terms.

    The framing is deliberately neutral about domain. An earlier version of
    this prompt opened with "extracted from a cluster of messages in a software
    engineering chat log", and the model returned "Software Engineering Chat
    Log" for every cluster it could not otherwise characterise. The prompt was
    handing over the generic answer and then being blamed for using it.
    """
    prompt = (
        "These keywords were extracted from one cluster of related documents:\n\n"
        + ", ".join(terms[:12])
        + (
            "\n\nExample passages from the cluster:\n\n"
            + "\n---\n".join(s[:300] for s in samples)
            if samples
            else ""
        )
        + "\n\nName this cluster's topic in at most five words. Use the "
        "keywords themselves wherever they are meaningful. If the keywords do "
        "not suggest a coherent topic, reply with exactly: UNCLEAR\n"
        "Reply with the label only — no quotes, no explanation, no full stop."
    )
    body = json.dumps(
        {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 24},
        }
    ).encode()

    try:
        req = urllib.request.Request(
            cfg.endpoint.rstrip("/") + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
            text = json.loads(r.read())["response"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        _log(f"  llm failed ({e}); falling back to terms")
        return None

    label = " ".join(text.strip().strip('"').split())
    # Giving the model an explicit way to decline is what makes the refusal
    # usable. Without UNCLEAR in the prompt it invents rather than abstains.
    if label.upper().startswith("UNCLEAR"):
        return None
    # A model that ignores "five words" and writes a paragraph has not produced
    # a label; the terms are better than a truncated sentence.
    return label if label and len(label) <= 60 else None


def label_clusters(
    cluster_terms: dict[int, list[str]],
    cluster_samples: dict[int, list[str]],
    cluster_sizes: dict[int, int],
    llm: LlmConfig | None = None,
    cache: dict[tuple[str, ...], str | None] | None = None,
) -> dict[int, dict]:
    """Label every cluster. Returns cluster id -> {label, terms, method}.

    Three things here exist purely because labelling is, by an enormous margin,
    the slowest stage of the pipeline. On the reference corpus the mathematics
    took 520 seconds and the labelling took **seven to eleven hours** — 98% of
    the runtime, at about 2.4 clusters a minute. Every instinct you have about
    speeding this pipeline up (a GPU, cuML, a bigger box) optimises the 1.3%.

    So:

    1. **The cache.** Facets overlap heavily — ``all`` contains every point in
       every other facet — so the same topic gets named repeatedly within one
       run. Keying on the term tuple makes the repeats free. Pass a dict in and
       persist it between runs and a rebuild becomes nearly free.
    2. **Skipping small clusters** (``llm.min_cluster_size``). Half of all
       clusters sit near the floor and cost the same as a 1,100-point one. The
       c-TF-IDF fallback for those is free and already good.
    3. **Parallelism** (``llm.workers``). The calls are independent and Ollama
       will happily serve several at once, so serial calling was leaving a
       straight multiple on the table.
    """
    cache = {} if cache is None else cache
    out: dict[int, dict] = {}

    def resolve(cid: int) -> tuple[int, dict]:
        terms = cluster_terms.get(cid, [])
        rec = {"terms": terms[:8], "label": None, "method": "c-tf-idf"}
        if not terms:
            return cid, rec

        too_small = llm and cluster_sizes.get(cid, 0) < llm.min_cluster_size
        if llm and not too_small:
            key = tuple(terms[:12])
            if key in cache:
                candidate = cache[key]
            else:
                candidate = llm_label(terms, cluster_samples.get(cid, []), llm)
                cache[key] = candidate
            if grounded(candidate, terms):
                rec["label"] = candidate
                rec["method"] = f"c-tf-idf+{llm.model}"

        if not rec["label"]:
            rec["label"] = terms_label(terms)
        return cid, rec

    ids = sorted(cluster_terms)
    t0 = time.perf_counter()
    if llm and llm.workers > 1:
        with ThreadPoolExecutor(max_workers=llm.workers) as pool:
            for cid, rec in pool.map(resolve, ids):
                out[cid] = rec
    else:
        for cid in ids:
            cid, rec = resolve(cid)
            out[cid] = rec

    if llm:
        named = sum(1 for r in out.values() if r["method"] != "c-tf-idf")
        _log(
            f"labelled {len(out)} clusters in {time.perf_counter() - t0:.0f}s "
            f"({named} from the model, {len(out) - named} from terms, "
            f"{len(cache)} distinct term sets)"
        )
    return out


def gather(
    ids: list[str],
    labels,
    text_by_id: dict[str, str],
    stop: set[str],
    n_samples: int = 3,
) -> tuple[dict[int, Counter], dict[int, list[str]]]:
    """Build the per-cluster pseudo-documents and pick example passages."""
    docs: dict[int, Counter] = {}
    samples: dict[int, list[str]] = {}
    for doc_id, cid in zip(ids, labels):
        cid = int(cid)
        if cid == -1:
            continue
        body = text_by_id.get(doc_id)
        if not body:
            continue
        docs.setdefault(cid, Counter()).update(tokens(body, stop))
        s = samples.setdefault(cid, [])
        if len(s) < n_samples:
            s.append(body)
    return docs, samples
