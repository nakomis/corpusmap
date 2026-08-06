# corpusmap

Cluster and label a text dataset: pgvector embeddings -> UMAP -> HDBSCAN ->
c-TF-IDF + optional LLM labels.

## Repository layout

| Path | What |
|---|---|
| `corpusmap/` | library + CLI (`db`, `embed`, `project`, `label`, `export`, `cli`) |
| `datasets/build_gutenberg.py` | builds the public-domain demo corpus |
| `tests/` | pytest; no database or dataset required |
| `data/` | generated dumps — gitignored, published as release assets |

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[fast,dev]'
docker compose up -d                     # pgvector on :5433
export DSN=postgresql://corpusmap:corpusmap@127.0.0.1:5433/corpusmap
.venv/bin/python -m pytest
```

Ollama models used: `nomic-embed-text` (768-dim embeddings) and `llama3.2:3b`
(label prettifying). Neither is required by the test suite.

## Things worth knowing before changing anything

- **`grounded()` in `label.py` is load-bearing.** It is what stops a small
  model from confidently naming every cluster it does not understand with the
  same fluent generic phrase. Do not relax it without re-reading its docstring.
- **Labelling is ~98% of runtime**, not the mathematics. Optimise there.
- **UMAP inter-cluster distance is meaningless.** Never add a feature that
  measures across the map.
- **Facets need separate fits** — UMAP projections do not compose.
- `random_state` is set deliberately, which forces UMAP single-threaded. A map
  that moves every rebuild cannot be talked about. Do not "fix" this.

## Testing

`pytest`. Aim for 70% minimum coverage. The suite is deliberately
database-free so it runs in seconds in CI.

## Architecture diagrams

Source: `docs/architecture/corpusmap.drawio` — SVG auto-regenerated on commit
by `.githooks/pre-commit`.

```bash
git config core.hooksPath .githooks
```
