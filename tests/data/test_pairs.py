"""Pair construction: identity exactness, inverse labels and determinism."""

from __future__ import annotations

import numpy as np

from image_fixtures import make_image
from retone.data.degradation import Magnitude
from retone.data.pairs import PairKind, PairPolicy, build_pair, choose_kind

TINY_POLICY = PairPolicy(
    identity_weight=0.0,
    synthetic_weight=1.0,
    mild_weight=0.0,
    synthetic_magnitude=Magnitude(1, 3),
    min_params=1,
    label_step=1,
    max_residual=1.0,
)


def test_identity_pair_leaves_the_image_untouched() -> None:
    image = make_image()
    pair = build_pair(image, np.random.default_rng(0), PairKind.IDENTITY, TINY_POLICY)
    assert np.array_equal(pair.input_image, image)
    assert pair.label.as_dict() == {name: 0 for name in pair.label.as_dict()}
    assert pair.residual == 0.0


def test_synthetic_label_is_the_inverse_degradation() -> None:
    image = make_image()
    pair = build_pair(image, np.random.default_rng(1), PairKind.SYNTHETIC, TINY_POLICY)
    degradation = pair.degradation.as_dict()
    label = pair.label.as_dict()
    for name, value in degradation.items():
        assert label[name] == -value
    assert label["sharpen_amount"] == 0
    assert pair.kind is PairKind.SYNTHETIC


def test_labels_round_to_the_policy_step() -> None:
    policy = PairPolicy(
        identity_weight=0.0,
        synthetic_weight=1.0,
        mild_weight=0.0,
        synthetic_magnitude=Magnitude(20, 30),
        label_step=5,
        max_residual=1.0,
    )
    pair = build_pair(make_image(), np.random.default_rng(2), PairKind.SYNTHETIC, policy)
    assert all(value % 5 == 0 for value in pair.label.as_dict().values())


def test_pairs_are_reproducible_for_a_seed() -> None:
    image = make_image()
    first = build_pair(image, np.random.default_rng(42), PairKind.SYNTHETIC, TINY_POLICY)
    second = build_pair(image, np.random.default_rng(42), PairKind.SYNTHETIC, TINY_POLICY)
    assert first.degradation == second.degradation
    assert np.array_equal(first.input_image, second.input_image)
    assert first.residual == second.residual


def test_kind_draws_follow_the_weights() -> None:
    identity_only = PairPolicy(identity_weight=1.0, synthetic_weight=0.0, mild_weight=0.0)
    rng = np.random.default_rng(0)
    assert {choose_kind(rng, identity_only) for _ in range(20)} == {PairKind.IDENTITY}


def test_policy_rejects_bad_weights() -> None:
    import pytest

    with pytest.raises(ValueError):
        PairPolicy(identity_weight=0.5, synthetic_weight=0.5, mild_weight=0.5)


def test_default_policy_requires_at_least_two_parameters() -> None:
    image = make_image()
    policy = PairPolicy()
    for seed in range(40):
        pair = build_pair(image, np.random.default_rng(seed), PairKind.SYNTHETIC, policy)
        non_zero = [name for name, value in pair.label.as_dict().items() if value != 0]
        assert len(non_zero) >= policy.min_params


def test_failed_pairs_come_from_the_small_fallback() -> None:
    # With an impossible residual limit the attempts all fail, so the pair must
    # be the small fallback (magnitudes 3..5) rather than a heavy synthetic one
    # handed back mid-loop.
    image = make_image()
    policy = PairPolicy(
        identity_weight=0.0,
        synthetic_weight=1.0,
        mild_weight=0.0,
        synthetic_magnitude=Magnitude(20, 70),
        min_params=2,
        max_residual=0.0,
    )
    pair = build_pair(image, np.random.default_rng(3), PairKind.SYNTHETIC, policy)
    assert not pair.accepted(policy)
    magnitudes = [abs(value) for value in pair.degradation.as_dict().values() if value != 0]
    assert magnitudes and all(3 <= value <= 5 for value in magnitudes)
