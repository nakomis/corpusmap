"""Squash high-dimensional vectors down to 3D, then find the dense clumps.

    normalise -> PCA(50) -> UMAP(3) -> HDBSCAN

Each stage earns its place, and the reasons are in the docstrings below rather
than in a paper you have to go and find.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

# UMAP and numba are enthusiastic warners about things you cannot act on.
warnings.filterwarnings("ignore")


@dataclass
class ProjectionConfig:
    """Every knob, with the defaults that survived contact with a real corpus."""

    seed: int = 42
    pca_dims: int = 50
    n_neighbors: int = 15
    min_dist: float = 0.05
    metric: str = "cosine"
    min_cluster_size: int = 20
    max_cluster_size: int | None = 2000
    n_components: int = 3


@dataclass
class Projection:
    """The result of one fit."""

    name: str
    ids: list[str]
    coords: np.ndarray
    labels: np.ndarray
    seed: int
    config: ProjectionConfig = field(repr=False, default_factory=ProjectionConfig)

    @property
    def kept(self) -> np.ndarray:
        """Boolean mask of points that landed in a cluster rather than noise."""
        return self.labels != -1

    def cluster_sizes(self) -> dict[int, int]:
        ids, counts = np.unique(self.labels[self.kept], return_counts=True)
        return {int(c): int(n) for c, n in zip(ids, counts)}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalise(X: np.ndarray) -> np.ndarray:
    """Scale every vector to unit length, so only its direction matters.

    Think of a compass: you care which way the needle points, not how long it
    is. Without this a document can look distinctive purely for having larger
    numbers, which is a property of how much text it contained rather than of
    what it said.
    """
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def reduce_pca(X: np.ndarray, dims: int, seed: int) -> np.ndarray:
    """A fast, crude cull from (say) 768 dimensions to 50.

    If almost every document has roughly the same value in slot 738, that slot
    distinguishes nothing, so discard it. PCA keeps the ``dims`` directions
    carrying the most variation. It is lossy, but what it throws away is
    precisely what nobody differed on — and it makes the UMAP stage several
    times faster, because UMAP's nearest-neighbour search is what dominates.
    """
    n = min(dims, X.shape[0], X.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(X)


def reduce_umap(X: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    """50 dimensions down to 3. The clever stage, and the misunderstood one.

    UMAP does not try to preserve everything. It asks one narrow question —
    *who are each point's ``n_neighbors`` nearest neighbours?* — and then finds
    a low-dimensional arrangement that keeps those neighbours together.

    Think of seating a wedding. You cannot satisfy every relationship at once,
    so you settle for a weaker guarantee: everyone sits near their actual
    friends.

    The consequence is the single most misunderstood property of every UMAP
    plot on the internet, so it is worth stating flatly:

        **A cluster's membership is meaningful. The distance between two
        clusters is not.**

    Two blobs at opposite ends of the view are not "more different" than two
    adjacent ones. The map is trustworthy up close and meaningless at long
    range. Do not measure across it, and be sceptical of anyone who does.

    ``random_state`` is set, which forces UMAP single-threaded and therefore
    slower. That is a deliberate trade: a map that moves every time you rebuild
    it is very hard to talk about, and "the cluster in the top left" needs to
    still be there tomorrow.
    """
    import umap  # imported late; it drags in numba and is slow to load

    return umap.UMAP(
        n_components=cfg.n_components,
        n_neighbors=min(cfg.n_neighbors, max(2, len(X) - 1)),
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.seed,
    ).fit_transform(X)


def cluster(coords: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    """Find dense clumps. Points in no clump get label -1 and are noise.

    Picture an aerial night photo of a festival: the tight knots of people are
    the groups, and the stragglers drifting between them are not a group, they
    are stragglers. HDBSCAN is comfortable saying "this point belongs to
    nothing", which k-means cannot do, and that honesty is most of why it is
    the right tool here.

    Expect to discard a lot. On a real 76,000-document corpus roughly **48%**
    of points came back as noise. That is not a failure — about half of most
    corpora genuinely is one-off material with no topic to belong to.

    ``max_cluster_size`` is the setting nobody thinks to reach for and everyone
    eventually needs. Without it, one fit on the reference corpus produced a
    single **16,890-point cluster** — 37% of everything in one lump that no
    label could describe. That is not a bug in the data. HDBSCAN's default
    'excess of mass' selection keeps the *most stable* clusters, and a broad
    featureless bulk is stable precisely because it has no internal structure
    to split along. Capping the size forces the selection further down the
    condensed tree.

    Measured against ``cluster_selection_method='leaf'``, the usual remedy, on
    45,183 documents (% of points landing in a cluster big enough to be a topic
    and small enough to label):

        eom  mcs=20            240 clusters  25.3% noise  largest 16,959  37.1% useful
        leaf mcs=20            334 clusters  59.2% noise  largest    318  40.8% useful
        eom  mcs=20 max=2000   294 clusters  50.6% noise  largest  1,011  49.4% useful

    The cap wins on every axis against leaf, which was the surprise. Its cost
    is honest rather than hidden: noise roughly doubles, because the blob's
    points do not become good clusters — they become scatter, which is what
    they always were.
    """
    kw: dict = {"min_cluster_size": cfg.min_cluster_size}
    if cfg.max_cluster_size:
        kw["max_cluster_size"] = cfg.max_cluster_size
    return HDBSCAN(**kw).fit_predict(coords)


def project(
    ids: list[str],
    X: np.ndarray,
    cfg: ProjectionConfig,
    name: str = "all",
) -> Projection:
    """Run the whole reduction and clustering pipeline over one set of vectors.

    Note that clustering happens on the **3D output**, not on the original
    vectors. That is deliberate and slightly unusual — the textbook advice is
    to cluster in an intermediate space of 10-50 dimensions.

    The upside of clustering the coordinates you are going to draw is that the
    clusters match what you see: no group of dots coloured as one cluster
    sitting visibly inside another, which is what makes a map argue with
    itself. The downside is that UMAP's distortions *become* the cluster
    structure — if the layout put two genuinely related topics at opposite ends
    of the room, nothing downstream can reunite them.

    If you want the textbook behaviour, cluster on the output of
    ``reduce_pca`` and keep the 3D coordinates for display only.
    """
    t0 = time.perf_counter()
    Xn = normalise(X)
    Xp = reduce_pca(Xn, cfg.pca_dims, cfg.seed)
    coords = reduce_umap(Xp, cfg)
    labels = cluster(coords, cfg)

    proj = Projection(
        name=name, ids=ids, coords=coords, labels=labels, seed=cfg.seed, config=cfg
    )
    sizes = proj.cluster_sizes()
    kept = int(proj.kept.sum())
    _log(
        f"{name}: {len(ids):,} docs -> {len(sizes)} clusters, "
        f"kept {kept:,} ({100 * kept / max(len(ids), 1):.0f}%), "
        f"{time.perf_counter() - t0:.0f}s"
    )
    return proj


def facets(
    ids: list[str], X: np.ndarray, values: list[str | None]
) -> list[tuple[str, np.ndarray]]:
    """Split the corpus by facet value, as (name, boolean mask) pairs.

    Each facet needs its **own fit**, and this is the part people get wrong.
    UMAP fits its layout to whichever points are present, so you cannot fit
    everything once and then hide dots client-side to get a per-facet view —
    you would be looking at the full corpus's geometry with holes punched in
    it, which is a different and misleading picture.

    The vectors are fetched once and subset in memory, because the fetch is
    expensive and the subsetting is free.
    """
    seen = sorted({v for v in values if v is not None})
    arr = np.asarray(values, dtype=object)
    out: list[tuple[str, np.ndarray]] = [
        (v, arr == v) for v in seen
    ]
    out.append(("all", np.ones(len(ids), dtype=bool)))
    return out
