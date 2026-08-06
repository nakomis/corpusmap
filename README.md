# corpusmap — Cluster and label any text dataset

## Support

If you find this useful, please consider buying me a coffee:

[![Donate with PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/donate?hosted_button_id=Q3BESC73EWVNN&custom=corpusmap)

Point it at a Postgres table with a text column and a
[pgvector](https://github.com/pgvector/pgvector) column, and it will tell you
what your corpus is *about* — as a set of named clusters and a 3D map you can
draw.

```
text -> embeddings -> normalise -> PCA -> UMAP(3) -> HDBSCAN -> c-TF-IDF -> LLM label
```

Nothing here is specific to one dataset. It came out of a system that maps
75,000 chat messages, but the demo corpus below is eight public-domain books,
and the tool cannot tell the difference.

## Table of Contents

<!-- toc -->

- [Quickstart](#quickstart)
- [Using it on your own data](#using-it-on-your-own-data)
  * [The four stages](#the-four-stages)
- [How it works, and what will surprise you](#how-it-works-and-what-will-surprise-you)
  * [Normalise, then PCA, then UMAP](#normalise-then-pca-then-umap)
  * [HDBSCAN, and the cap nobody mentions](#hdbscan-and-the-cap-nobody-mentions)
  * [Labels come from c-TF-IDF, not from the model](#labels-come-from-c-tf-idf-not-from-the-model)
  * [The grounding check](#the-grounding-check)
  * [Labelling is 98% of the runtime](#labelling-is-98%25-of-the-runtime)
  * [Facets need their own fit](#facets-need-their-own-fit)
- [The demo dataset](#the-demo-dataset)
  * [Download](#download)
- [Repository Layout](#repository-layout)
- [Testing](#testing)
- [Architecture Diagrams](#architecture-diagrams)
- [Licence](#licence)
- [Support](#support)

<!-- tocstop -->

## Quickstart

Five commands, about ten minutes, no cloud account and no API key. You need
Docker, Python 3.10+, and [Ollama](https://ollama.com).

```bash
git clone git@github.com:nakomis/corpusmap.git && cd corpusmap
python3 -m venv .venv && .venv/bin/pip install -e '.[fast]'

# A throwaway Postgres with pgvector on port 5433
docker compose up -d

# ~15,500 passages from eight out-of-copyright books
ollama pull nomic-embed-text && ollama pull llama3.2:3b
export DSN=postgresql://corpusmap:corpusmap@127.0.0.1:5433/corpusmap
.venv/bin/python datasets/build_gutenberg.py --dsn "$DSN" --reset

# Embed, project, cluster, label
.venv/bin/corpusmap embed --dsn "$DSN" --model nomic-embed-text --workers 8
.venv/bin/corpusmap run   --dsn "$DSN" --facet-col category \
                          --out build --prefix gutenberg --llm llama3.2:3b
```

Or skip the embedding entirely and
[download the pre-built dataset](#the-demo-dataset).

The result is `build/gutenberg-*.json` — one file per facet plus a manifest.
The largest clusters in `gutenberg-all.json` on a reference run:

```
   980  thou, unto, thee
   845  elizabeth, darcy, bennet
   816  Whaling and Whalemen Culture
   737  Nautical Mathematics and Slang
   402  holmes, sherlock, said
   361  Jane Austen Literary Analysis
   276  Jules Verne's Nautilus Adventure
   234  stewpan, simmer, put
   214  Birds and Pigeon Breeds
   136  Plant Breeding and Sterility
```

Which is the point: the recipes separated from the marine biology without being
told that either existed.

## Using it on your own data

The only requirement is a table with an id, some text, and a vector column.
Every name is configurable:

```bash
corpusmap run \
  --dsn "$DSN" \
  --table articles \
  --id-col article_id \
  --text-col body \
  --vector-col embedding \
  --facet-col source \
  --where "published_at > '2024-01-01'" \
  --llm llama3.2:3b
```

If your vector column is already populated by something upstream, skip
`corpusmap embed` and go straight to `run`.

### The four stages

| Command | Does | Typical share of runtime |
|---|---|---|
| `corpusmap embed` | text -> vectors, via Ollama. Resumable. | varies |
| `corpusmap project` | vectors -> 3D coordinates + cluster ids | ~1% |
| `corpusmap label` | cluster ids -> English names | **~98%** |
| `corpusmap run` | project then label, in one go | |

`project` and `label` are separate on purpose. The fit is cached to
`build/<prefix>-state.npz`, so you can re-run labelling with a different stop
list or a different model without re-fitting anything.

## How it works, and what will surprise you

### Normalise, then PCA, then UMAP

**Normalise** scales every vector to unit length so only its *direction*
matters — a compass, where you care which way the needle points, not how long
it is. Without this a document looks distinctive for having larger numbers,
which is a fact about its length rather than its meaning.

**PCA** culls 768 dimensions down to 50. If nearly every document has the same
value in slot 738, that slot distinguishes nothing. Lossy, but it only discards
what nobody differed on — and it makes UMAP several times faster.

**UMAP** takes 50 down to 3. It does not preserve everything; it asks one
narrow question — *who are each point's 15 nearest neighbours?* — and finds an
arrangement that keeps those neighbours together. Think of seating a wedding:
you cannot satisfy every relationship, so you settle for "everyone sits near
their actual friends".

> **The one thing to take away.** A cluster's *membership* is meaningful. The
> *distance between two clusters is not.* Two blobs at opposite ends of the
> view are not "more different" than two adjacent ones. The map is trustworthy
> up close and meaningless at long range. This is true of every UMAP plot you
> have ever seen, including the ones with confident arrows drawn across them.

### HDBSCAN, and the cap nobody mentions

HDBSCAN finds dense clumps — an aerial night photo of a festival, where tight
knots of people are groups and the stragglers between them are not. Crucially
it is willing to say "this point belongs to nothing", which k-means cannot.

**Expect to discard a lot.** 42% of the demo corpus comes back as noise; the
reference chat corpus was 48%. That is not a failure. About half of most
corpora genuinely is one-off material with no topic to belong to.

`--max-cluster-size` is the setting you will not think to reach for and will
eventually need. Without it, one fit on the reference corpus produced a single
**16,890-point cluster** — 37% of everything in one lump that no label could
describe. That is not a bug in the data: HDBSCAN's default *excess of mass*
selection keeps the most **stable** clusters, and a broad featureless bulk is
stable precisely because it has no internal structure to split along.

Measured on 45,183 documents — % of points landing in a cluster big enough to
be a topic and small enough to label:

| Selection | Clusters | Noise | Largest | Useful |
|---|---|---|---|---|
| `eom`, mcs=20 | 240 | 25.3% | 16,959 | 37.1% |
| `leaf`, mcs=20 | 334 | 59.2% | 318 | 40.8% |
| `eom`, mcs=20, **max=2000** | 294 | 50.6% | **1,011** | **49.4%** |

The cap beats `leaf` — the usual remedy — on every axis. Its cost is honest
rather than hidden: noise roughly doubles, because the blob's points do not
become good clusters, they become scatter, which is what they always were.

### Labels come from c-TF-IDF, not from the model

This is backwards from most people's instinct, and it matters.

**c-TF-IDF** asks "which terms are frequent in this cluster and rare in the
rest of the corpus" — exactly the question a label answers. It is
deterministic, free, and runs over the whole corpus in seconds. An LLM asked to
name a cluster has to be shown a sample, and a sample of a 3,000-document
cluster is a guess about the other 2,990.

So the LLM is a **second pass that turns terms into readable English**. It
renames; it does not decide. Switch it off and you get raw terms — uglier, but
never wrong about what the cluster contains.

### The grounding check

A small model asked to name a cluster it does not understand **does not say
so**. It emits a confident generic phrase. Measured over 1,460 clusters with
`llama3.2:3b`, always in the same direction:

```
ddr3, sata, fpga     ->  "Software Engineering Chat Log"
great, nah, mornin   ->  "Software engineering meeting discussion"
spacex, spcx, hedge  ->  "Software Engineering Chat Log"
```

None of those is a paraphrase. They are inventions, and each is *worse* than
the terms it replaced, because it reads as authoritative whilst saying nothing.

So a label is only kept if at least one of the cluster's top terms actually
appears in it. Long terms match on their first four characters at a word
boundary (so "testing" grounds "tests"); short terms must match as whole words.
It costs nothing, needs no second model, and turns an unusable labelling pass
into a usable one.

You can watch it working in the demo output. The 980-passage cluster whose top
terms are `thou, unto, thee` kept its terms rather than a label — the model's
suggestion was not grounded. That cluster is Marcus Aurelius, and it formed on
*register* rather than subject, which is a real and interesting property of the
corpus that a fluent invented label would have papered over.

### Labelling is 98% of the runtime

On the reference corpus: 520 seconds of mathematics, **seven to eleven hours**
of a 3B model writing five-word labels, one cluster at a time, 1,634 times.
Every instinct you have about speeding this up — a GPU, cuML, a bigger box —
optimises the 1.3%.

Three things here address that, cheapest first:

1. **The term-set cache.** Facets overlap heavily (`all` contains every point
   in all the others), so the same topic gets named repeatedly within one run.
   Keying on the term tuple makes the repeats free. In the demo run, 145
   distinct term sets covered 145 clusters across 8 facets.
2. **`--llm-min-cluster-size`.** Half of all clusters sit near the floor and
   cost the same as a 1,100-point one. Fall back to terms for those.
3. **`--llm-workers`.** The calls are independent; serial calling leaves a
   straight multiple on the table.

### Facets need their own fit

`--facet-col` splits the corpus by a column and fits **each value separately**.
This is not tidiness. UMAP fits its layout to whichever points are present, so
you cannot fit everything once and hide dots client-side to get a per-facet
view — you would be looking at the full corpus's geometry with holes punched in
it. The vectors are fetched once and subset in memory, because the fetch is
expensive and the subsetting is free.

## The demo dataset

Eight out-of-copyright books from Project Gutenberg, split into ~15,500
passages of up to 600 characters, embedded with `nomic-embed-text` (768
dimensions).

| Category | Passages | Books |
|---|---|---|
| `fiction-adventure` | 4,078 | *Moby Dick*, *Twenty Thousand Leagues under the Sea* |
| `science-travel` | 2,500 | *The Voyage of the Beagle* |
| `reference-cookery` | 2,500 | *The Book of Household Management* |
| `science-biology` | 2,073 | *On the Origin of Species* |
| `fiction-social` | 1,942 | *Pride and Prejudice* |
| `fiction-detective` | 1,505 | *The Adventures of Sherlock Holmes* |
| `philosophy` | 936 | *Meditations* |

Eight books rather than one because a single novel clusters badly — one author,
one register, one subject, so the map comes out a single smear. A deliberately
mixed shelf gives clusters you can check by eye: if the recipes do not separate
from the marine biology, something is wrong with your pipeline and not with the
world.

Passages rather than sentences because a sentence is usually too short to have
a topic. "He said nothing." belongs to no cluster in any corpus.

### Download

Both dumps are on the [latest release](https://github.com/nakomis/corpusmap/releases/latest).

| File | Size | Contains |
|---|---|---|
| `gutenberg-corpus.dump` | 59 MB | passages **and** 768-dim embeddings |
| `gutenberg-text-only.dump` | 2.4 MB | passages only — embed with your own model |

```bash
docker compose up -d
pg_restore -d "$DSN" --no-owner --no-acl gutenberg-corpus.dump
corpusmap run --dsn "$DSN" --facet-col category --llm llama3.2:3b
```

Or rebuild it yourself from source — `datasets/build_gutenberg.py` downloads
the books directly from Project Gutenberg.

Everything in it is public domain, and Project Gutenberg's own licence header
and footer are stripped (left in, they form their own large, perfectly coherent
cluster about the Project Gutenberg Literary Archive Foundation — amusing once,
useless twice). See [datasets/SOURCES.md](datasets/SOURCES.md).

## Repository Layout

| Path | What |
|---|---|
| `corpusmap/` | the library and CLI |
| `corpusmap/project.py` | normalise, PCA, UMAP, HDBSCAN |
| `corpusmap/label.py` | c-TF-IDF, the LLM pass, the grounding check |
| `datasets/build_gutenberg.py` | builds the demo corpus from Project Gutenberg |
| `tests/` | pytest suite; no database or dataset needed |
| `docker-compose.yml` | throwaway Postgres + pgvector on port 5433 |

## Testing

```bash
.venv/bin/python -m pytest
```

## Architecture Diagrams

`docs/architecture/corpusmap.drawio` is the source for the architecture
diagram. The SVG is auto-regenerated on commit by `.githooks/pre-commit`.

To activate the hook after cloning:

```bash
git config core.hooksPath .githooks
```

## Licence

Code and documentation: [CC0 1.0](LICENSE) — public domain, do what you like.

The demo corpus is built from works that are out of copyright in the UK and the
US.

## Support

If you find this useful, please consider buying me a coffee:

[![Donate with PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/donate?hosted_button_id=Q3BESC73EWVNN&custom=corpusmap)
