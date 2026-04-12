"""cpu_variants: GUI CPU style list matches sweep TOML rows."""

from __future__ import annotations

from motley_crews_play.cpu_variants import CPU_VARIANTS, default_cpu_variant_index, policy_for_variant_index


def test_cpu_variants_count() -> None:
    assert len(CPU_VARIANTS) == 12


def test_default_index_is_default_like() -> None:
    assert CPU_VARIANTS[default_cpu_variant_index()].key == "default_like"


def test_policy_for_each_index() -> None:
    for i in range(len(CPU_VARIANTS)):
        p = policy_for_variant_index(i)
        assert p.weights.vp_scale > 0


def test_barbarian_emphasis_is_hard() -> None:
    barb = next(v for v in CPU_VARIANTS if v.key == "barbarian_emphasis")
    assert barb.difficulty == "hard"


def test_melee_aggressive_is_hard() -> None:
    m = next(v for v in CPU_VARIANTS if v.key == "melee_aggressive")
    assert m.difficulty == "hard"
