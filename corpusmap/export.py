"""Write the finished map out as JSON.

One file per facet plus a manifest, which is the shape a viewer wants: it can
load the manifest to build a facet switcher and then fetch only the projection
being looked at.

No edges are emitted, and that is a considered omission. UMAP already encodes
similarity as *position* — that is the entire point of the algorithm — so
drawing edges largely redraws what proximity already shows, and at 40,000+
nodes they render as grey fog. On the reference corpus dropping them took each
artefact from about 13 MB to about 1.5 MB. If you want edges within one
cluster, compute them on demand in the viewer for the handful of points the
user has actually selected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .project import Projection


def projection_doc(
    proj: Projection,
    labels: dict[int, dict] | None = None,
    facet_by_id: dict[str, str] | None = None,
    precision: int = 3,
) -> dict:
    labels = labels or {}
    sizes = proj.cluster_sizes()
    keep = proj.kept

    nodes = []
    for i in range(len(proj.ids)):
        if not keep[i]:
            continue
        node = {
            "x": round(float(proj.coords[i][0]), precision),
            "y": round(float(proj.coords[i][1]), precision),
            "z": round(float(proj.coords[i][2]), precision),
            "c": int(proj.labels[i]),
            "id": proj.ids[i],
        }
        if facet_by_id:
            node["f"] = facet_by_id.get(proj.ids[i])
        nodes.append(node)

    return {
        "name": proj.name,
        "seed": proj.seed,
        "min_cluster_size": proj.config.min_cluster_size,
        "max_cluster_size": proj.config.max_cluster_size,
        "total": len(proj.ids),
        "kept": int(keep.sum()),
        "dropped_noise": int((~keep).sum()),
        "nodes": nodes,
        "clusters": [
            {
                "id": cid,
                "size": size,
                "label": labels.get(cid, {}).get("label"),
                "terms": labels.get(cid, {}).get("terms", []),
            }
            for cid, size in sorted(sizes.items())
        ],
    }


def write_docs(
    out_dir: str | Path,
    prefix: str,
    docs: list[dict],
    text_by_id: dict[str, str],
    meta_by_id: dict[str, list],
    meta_names: list[str],
    snippet_chars: int = 320,
) -> Path:
    """Write the hover-card text table, shared across every facet.

    Deduplicated deliberately. A document appears in its own facet's projection
    *and* in ``all``, so storing the text per node would carry most of the
    corpus twice; keying it by document id instead means the second appearance
    costs a 4-byte index.

    Snippets are truncated because this is a hover card, not a reader. The
    passages in the demo corpus run to 600 characters and the full text of
    ~12,000 of them would roughly double the size of a self-contained artefact
    that is otherwise a couple of hundred kilobytes.
    """
    wanted: set[str] = set()
    for doc in docs:
        for node in doc.get("nodes", []):
            wanted.add(node["id"])

    table: dict[str, list] = {}
    for doc_id in wanted:
        body = text_by_id.get(doc_id) or ""
        snippet = body[:snippet_chars]
        if len(body) > snippet_chars:
            snippet = snippet.rstrip() + "…"
        table[doc_id] = [snippet, *meta_by_id.get(doc_id, [])]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}-docs.json"
    path.write_text(
        json.dumps({"meta_names": meta_names, "docs": table}, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def write(
    out_dir: str | Path,
    prefix: str,
    docs: list[dict],
    extra: dict | None = None,
) -> Path:
    """Write each projection and a manifest. Returns the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"prefix": prefix, "projections": [], **(extra or {})}
    for doc in docs:
        path = out / f"{prefix}-{doc['name']}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        manifest["projections"].append(
            {
                "name": doc["name"],
                "file": path.name,
                "nodes": doc["kept"],
                "clusters": len(doc["clusters"]),
                "mb": round(os.path.getsize(path) / 1e6, 2),
            }
        )

    manifest_path = out / f"{prefix}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
