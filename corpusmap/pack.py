"""Pack projections into a single self-contained HTML viewer.

The artefact has to open from `file://` with no network access of any kind,
which rules out fetching the data alongside it: browsers block `fetch()` from
`file://`. So the data is inlined — and raw JSON is bigger than it needs to be
when you are inlining every facet at once.

Coordinates are therefore quantised to int16 over each projection's own
bounding box. UMAP output is a *relative* layout with no meaningful units, so
1/65,536 of the extent is far below anything the eye or the renderer can
resolve — this is a lossless change as far as anyone looking at it is
concerned. That takes a node from ~40 bytes of JSON to 8 bytes of binary:

    int16 x, int16 y, int16 z, uint16 cluster, uint32 doc

35,000 nodes then cost ~420 KB raw and ~560 KB as base64, so a whole run's
worth of facets inlines comfortably.

The trailing ``doc`` is an index into one shared text table rather than the
text itself. A document appears both in its own facet's projection and in
``all``, so inlining the passage per node would carry most of the corpus
twice; an index costs four bytes on the second appearance.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

# Assembled rather than written literally, so that this line does not itself
# count as an occurrence of the token. See `_inject`.
TOKEN = "/*__" + "DATA__*/{}"


def pack_projection(doc: dict, doc_index: dict[str, int] | None = None) -> dict | None:
    """Quantise one projection document into an inlinable payload.

    ``doc_index`` maps document id -> row in the shared text table. Nodes whose
    id is absent get the sentinel 0xFFFFFFFF and simply show no hover card,
    which is the right outcome for a document whose text went missing between
    projecting and packing.
    """
    nodes = doc.get("nodes") or []
    if not nodes:
        return None
    doc_index = doc_index or {}
    NO_DOC = 0xFFFFFFFF

    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    zs = [n["z"] for n in nodes]
    lo = [min(xs), min(ys), min(zs)]
    hi = [max(xs), max(ys), max(zs)]
    # Guard the degenerate axis: a projection could in principle be flat in one
    # dimension, and dividing by a zero span would produce NaNs for every node.
    span = [(h - l) or 1.0 for l, h in zip(lo, hi)]

    buf = bytearray()
    for n in nodes:
        q = [
            int(round((v - l) / s * 65535)) - 32768
            for v, l, s in zip((n["x"], n["y"], n["z"]), lo, span)
        ]
        buf += struct.pack(
            "<hhhHI",
            max(-32768, min(32767, q[0])),
            max(-32768, min(32767, q[1])),
            max(-32768, min(32767, q[2])),
            # Cluster -1 (noise) is already dropped upstream, but clamp rather
            # than trusting it — a negative id would wrap to 65535 and silently
            # merge with a real cluster.
            max(0, min(65535, n["c"])),
            doc_index.get(n["id"], NO_DOC),
        )

    clusters = {c["id"]: c for c in doc.get("clusters", [])}
    return {
        "name": doc["name"],
        "total": doc.get("total", len(nodes)),
        "kept": doc.get("kept", len(nodes)),
        "n": len(nodes),
        "data": base64.b64encode(bytes(buf)).decode(),
        "clusters": [
            {
                "id": int(k),
                "size": v.get("size", 0),
                "label": v.get("label"),
                "terms": v.get("terms"),
            }
            for k, v in sorted(clusters.items())
        ],
    }


def _inject(template: str, bundle: dict) -> str:
    """Substitute the data placeholder — exactly once, or refuse.

    "Exactly once", not merely present. In the viewer this was ported from, a
    comment documenting the token was enough to make `str.replace` inline the
    whole payload twice. The failure is silent: the page still works, because
    the first copy lands inside a comment, and the only symptom is a bundle of
    double the expected size. That shipped at 53 MB before anyone noticed.
    """
    seen = template.count(TOKEN)
    if seen != 1:
        raise SystemExit(
            f"viewer.html must contain the data placeholder exactly once, found {seen}"
        )
    return template.replace(TOKEN, json.dumps(bundle, separators=(",", ":")))


def build(
    docs: list[dict],
    out_path: str | Path,
    corpus: str = "corpus",
    template: str | Path | None = None,
    doc_table: dict | None = None,
) -> Path:
    """Write a single self-contained HTML file for these projections.

    ``doc_table`` is the ``{"meta_names": [...], "docs": {id: [...]}}`` mapping
    written by ``export.write_docs``. Omit it and the map still works — it just
    has no hover cards.
    """
    tpl = Path(template) if template else Path(__file__).with_name("viewer.html")

    # The shared text table is emitted as a list, so each node can carry a
    # 4-byte row index rather than its document id as a string.
    meta_names: list[str] = []
    rows: list[list] = []
    doc_index: dict[str, int] = {}
    if doc_table:
        meta_names = doc_table.get("meta_names", [])
        for doc_id, record in doc_table.get("docs", {}).items():
            doc_index[doc_id] = len(rows)
            rows.append(record)

    packed = {}
    for doc in docs:
        p = pack_projection(doc, doc_index)
        if p:
            packed[doc["name"]] = p

    if not packed:
        raise SystemExit("nothing to pack — every projection was empty")

    bundle = {
        "corpus": corpus,
        "projections": packed,
        "meta_names": meta_names,
        "docs": rows,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_inject(tpl.read_text(encoding="utf-8"), bundle), encoding="utf-8")
    return out


def build_from_dir(
    out_dir: str | Path,
    prefix: str,
    out_path: str | Path | None = None,
    corpus: str | None = None,
) -> Path:
    """Pack the JSON artefacts a previous run left in `out_dir`."""
    src = Path(out_dir)
    manifest_path = src / f"{prefix}-manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no {manifest_path} — run `corpusmap project` first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = []
    for entry in manifest.get("projections", []):
        path = src / entry["file"]
        if path.exists():
            docs.append(json.loads(path.read_text(encoding="utf-8")))

    # Optional: a run with --no-viewer, or an older build, will not have one.
    docs_path = src / f"{prefix}-docs.json"
    doc_table = (
        json.loads(docs_path.read_text(encoding="utf-8")) if docs_path.exists() else None
    )

    return build(
        docs,
        out_path or (src / f"{prefix}.html"),
        corpus=corpus or prefix,
        doc_table=doc_table,
    )
