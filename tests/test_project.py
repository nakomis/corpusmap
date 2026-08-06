"""Tests for the reduction and clustering stage.

These use small synthetic blobs rather than the demo corpus, so they run in
seconds and do not need a database or a downloaded dataset.
"""

import numpy as np
import pytest

from corpusmap.project import (
    ProjectionConfig,
    cluster,
    facets,
    normalise,
    project,
    reduce_pca,
)


def blobs(n_per=60, k=3, dims=32, seed=0):
    """k well-separated clusters in `dims` dimensions."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=8.0, size=(k, dims))
    return np.vstack(
        [c + rng.normal(scale=0.3, size=(n_per, dims)) for c in centres]
    ).astype(np.float32)


class TestNormalise:
    def test_every_row_becomes_unit_length(self):
        X = np.random.default_rng(0).normal(size=(50, 16))
        assert np.allclose(np.linalg.norm(normalise(X), axis=1), 1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        X = np.zeros((3, 8))
        assert np.isfinite(normalise(X)).all()

    def test_direction_is_preserved(self):
        X = np.array([[3.0, 4.0]])
        assert np.allclose(normalise(X), [[0.6, 0.8]])


class TestReducePca:
    def test_output_shape(self):
        assert reduce_pca(blobs(), dims=10, seed=0).shape == (180, 10)

    def test_dims_clamped_to_available(self):
        # Asking for more components than you have rows or columns is a
        # perfectly reasonable thing to do on a tiny corpus and must not crash.
        X = blobs(n_per=3, k=2, dims=8)
        assert reduce_pca(X, dims=50, seed=0).shape[1] <= 6

    def test_deterministic_under_a_fixed_seed(self):
        X = blobs()
        assert np.allclose(reduce_pca(X, 10, seed=42), reduce_pca(X, 10, seed=42))


class TestCluster:
    def test_finds_separated_groups(self):
        coords = np.vstack(
            [
                np.random.default_rng(s).normal(loc=c, scale=0.2, size=(40, 3))
                for s, c in enumerate([0, 20, 40])
            ]
        )
        labels = cluster(coords, ProjectionConfig(min_cluster_size=10))
        assert len({int(x) for x in labels if x != -1}) == 3

    def test_noise_is_labelled_minus_one(self):
        # One tight blob plus scattered outliers: the outliers must be allowed
        # to belong to nothing, which is most of why HDBSCAN is the right tool.
        rng = np.random.default_rng(0)
        coords = np.vstack(
            [rng.normal(scale=0.2, size=(60, 3)), rng.normal(scale=30, size=(20, 3))]
        )
        labels = cluster(coords, ProjectionConfig(min_cluster_size=10))
        assert (labels == -1).any()

    def test_max_cluster_size_splits_a_single_blob(self):
        # The behaviour the cap exists for: without it one broad featureless
        # bulk is accepted whole, because a blob with no internal structure is
        # maximally *stable* by HDBSCAN's default selection.
        coords = np.random.default_rng(0).normal(scale=1.0, size=(400, 3))
        uncapped = cluster(coords, ProjectionConfig(min_cluster_size=20,
                                                    max_cluster_size=None))
        capped = cluster(coords, ProjectionConfig(min_cluster_size=20,
                                                  max_cluster_size=50))
        biggest = lambda l: max(  # noqa: E731
            [int((l == c).sum()) for c in set(l) if c != -1] or [0]
        )
        assert biggest(capped) <= biggest(uncapped)


class TestProject:
    def test_end_to_end_shape_and_determinism(self):
        X = blobs()
        ids = [str(i) for i in range(len(X))]
        cfg = ProjectionConfig(pca_dims=10, min_cluster_size=10, max_cluster_size=None)
        a = project(ids, X, cfg, name="t")
        b = project(ids, X, cfg, name="t")
        assert a.coords.shape == (len(X), 3)
        # A map that moves every rebuild is impossible to talk about, so the
        # fixed seed has to actually hold.
        assert np.allclose(a.coords, b.coords)
        assert (a.labels == b.labels).all()

    def test_kept_mask_excludes_noise(self):
        X = blobs()
        cfg = ProjectionConfig(pca_dims=10, min_cluster_size=10)
        proj = project([str(i) for i in range(len(X))], X, cfg)
        assert proj.kept.sum() == (proj.labels != -1).sum()
        assert sum(proj.cluster_sizes().values()) == proj.kept.sum()


class TestFacets:
    def test_one_mask_per_value_plus_all(self):
        ids = ["a", "b", "c", "d"]
        X = np.zeros((4, 4))
        out = facets(ids, X, ["x", "y", "x", None])
        names = [n for n, _ in out]
        assert names == ["x", "y", "all"]
        assert dict(out)["x"].tolist() == [True, False, True, False]
        # "all" includes the row with no facet value — it is still a document.
        assert dict(out)["all"].all()

    @pytest.mark.parametrize("values", [[None, None], []])
    def test_no_facet_values_still_yields_all(self, values):
        out = facets(list(range(len(values))), np.zeros((len(values), 2)), values)
        assert [n for n, _ in out] == ["all"]


class TestFetchOrdering:
    """The fetch must impose an order, or the fixed seed guarantees nothing.

    UMAP and HDBSCAN both depend on the order their input arrives in. Without
    an ORDER BY, Postgres returns rows in physical heap order — which is not id
    order, and is not stable, because `corpusmap embed` UPDATEs every row and
    autovacuum can reshuffle them afterwards.

    Measured on the demo corpus before this was fixed: the same 15,534
    documents, same seed and same code gave 53 clusters / 8,974 kept on one run
    and 51 / 9,553 on another. Both runs looked entirely healthy.

    These assert against the SQL `db.fetch` actually issues, by standing a fake
    connection in front of psycopg. Rebuilding the query string in the test
    would pass even if the clause were deleted from the source, which is the
    one thing this must never do.
    """

    @staticmethod
    def _captured_sql(monkeypatch, **corpus_kw):
        import corpusmap.db as db

        seen = {}

        class FakeCursor:
            def execute(self, sql):
                seen["sql"] = " ".join(sql.split())

            def fetchall(self):
                # One row, so fetch() gets past its "no rows" guard. Three
                # columns because fetch reads r[2] whenever a facet column is
                # configured — a two-column fake passes the no-facet tests and
                # then IndexErrors on the faceted one.
                return [("1", [0.0, 1.0], "cat")]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db.psycopg, "connect", lambda dsn: FakeConn())
        db.fetch(db.Corpus(dsn="postgresql:///x", **corpus_kw))
        return seen["sql"]

    def test_fetch_orders_by_id(self, monkeypatch):
        sql = self._captured_sql(monkeypatch)
        assert sql.upper().endswith("ORDER BY ID")

    def test_ordering_uses_the_configured_id_column(self, monkeypatch):
        sql = self._captured_sql(monkeypatch, id_col="doc_pk")
        assert sql.endswith("ORDER BY doc_pk")

    def test_ordering_survives_facets_and_a_where_clause(self, monkeypatch):
        sql = self._captured_sql(
            monkeypatch, facet_col="category", where="seq > 10"
        )
        assert "category::text" in sql and "seq > 10" in sql
        assert sql.endswith("ORDER BY id")
