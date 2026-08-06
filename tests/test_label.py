"""Tests for the labelling stage.

The grounding check gets the most attention here, because it is the piece
doing the real work — it is what stands between you and 1,600 clusters all
confidently named after whatever generic phrase your model likes best.
"""

from collections import Counter

from corpusmap.label import (
    STOP_WORDS,
    ctfidf,
    grounded,
    is_junk,
    terms_label,
    tokens,
)


class TestGrounded:
    def test_label_mentioning_a_top_term_is_kept(self):
        assert grounded("Whaling voyages and harpoons", ["whaling", "harpoon", "ship"])

    def test_prefix_match_survives_inflection(self):
        # "testing" grounds a label saying "tests" — the whole reason the check
        # is a prefix match rather than equality.
        assert grounded("Unit tests for the release", ["testing"])
        assert grounded("Whaling voyages", ["whales"] and ["whaling"])

    def test_case_is_ignored(self):
        assert grounded("Cognito user pools", ["cognito", "pools"])

    def test_confident_generic_invention_is_rejected(self):
        # The observed failure mode: a small model names a cluster it does not
        # understand with a fluent phrase that shares nothing with the evidence.
        assert not grounded(
            "Software Engineering Chat Log", ["ddr3", "sata", "fpga", "pcie"]
        )
        assert not grounded(
            "Software Engineering Chat Log", ["spacex", "spcx", "hedge"]
        )

    def test_short_terms_must_match_as_whole_words(self):
        # A four-character prefix of a short word matches far too much, and a
        # plain substring test is worse: "sata" must not ground "Satanic".
        assert not grounded("Satanic verses", ["sata"])
        assert grounded("Sata interfaces", ["sata"])

    def test_long_terms_match_at_a_word_boundary_only(self):
        # "harpoon" -> stem "harp" must not be satisfied by a word that merely
        # contains those letters somewhere in the middle.
        assert not grounded("Sharpening the blade", ["harpoon"])
        assert grounded("Harpoons and rope", ["harpoon"])

    def test_empty_inputs_are_not_grounded(self):
        assert not grounded(None, ["ship"])
        assert not grounded("", ["ship"])
        assert not grounded("Ships", [])


class TestIsJunk:
    def test_long_identifiers_are_junk(self):
        assert is_junk("a" * 19)

    def test_hash_like_tokens_are_junk(self):
        assert is_junk("f3a9b2c7d1")

    def test_ordinary_words_are_not_junk(self):
        assert not is_junk("harpoon")
        assert not is_junk("pgvector")

    def test_word_with_a_version_number_survives(self):
        # "llama3" is a topic; "a1b2c3d4" is not. Both mix letters and digits,
        # so the rule is about how many digits, not whether there are any.
        assert not is_junk("llama3")
        assert not is_junk("python3")


class TestTokens:
    def test_stop_words_are_dropped(self):
        assert list(tokens("the and but harpoon", STOP_WORDS)) == ["harpoon"]

    def test_short_tokens_are_dropped(self):
        assert "ab" not in list(tokens("ab harpoon", STOP_WORDS))

    def test_case_is_folded(self):
        assert list(tokens("Harpoon HARPOON", STOP_WORDS)) == ["harpoon", "harpoon"]


class TestCtfidf:
    def test_term_common_here_and_rare_elsewhere_wins(self):
        docs = {
            0: Counter({"harpoon": 10, "ship": 5, "water": 5}),
            1: Counter({"flour": 10, "butter": 5, "water": 5}),
            2: Counter({"finch": 10, "beak": 5, "water": 5}),
        }
        top = ctfidf(docs, top_n=3)
        # "water" is in every cluster, so it must not lead any of them, even
        # though it is frequent in all three.
        assert top[0][0] == "harpoon"
        assert top[1][0] == "flour"
        assert top[2][0] == "finch"
        assert "water" not in top[0][:1]

    def test_singleton_terms_dropped_from_large_clusters(self):
        # A term appearing once in a 300-term cluster is noise, not a topic.
        docs = {
            0: Counter({"harpoon": 200, "ship": 100, "fluke": 1}),
            1: Counter({"flour": 200, "butter": 100}),
        }
        assert "fluke" not in ctfidf(docs, top_n=10)[0]

    def test_singleton_terms_kept_in_small_clusters(self):
        # In a small cluster a single occurrence may be all the evidence there is.
        docs = {0: Counter({"harpoon": 2, "fluke": 1}), 1: Counter({"flour": 2})}
        assert "fluke" in ctfidf(docs, top_n=10)[0]


class TestTermsLabel:
    def test_joins_top_three(self):
        assert terms_label(["a", "b", "c", "d"]) == "a, b, c"

    def test_none_when_no_terms(self):
        assert terms_label([]) is None
