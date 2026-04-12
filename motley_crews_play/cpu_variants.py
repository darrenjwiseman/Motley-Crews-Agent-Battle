"""
Selectable CPU opponent styles for the Pygame client.

Weights match ``config/sweep.example.wide_separation.toml``. Difficulty tiers
(``easy`` / ``normal`` / ``hard``) follow ``config/sweep_out/report_wide_master.md``
(Wilson lower bound vs stock ``ScriptedHeuristicPolicy``), with
``barbarian_emphasis`` placed in **hard** as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

from motley_crews_play.policies import ParameterizedHeuristicPolicy, heuristic_weights_from_spec


@dataclass(frozen=True, slots=True)
class CpuVariantDef:
    """One sweep-style variant row for the GUI CPU picker."""

    key: str
    difficulty: str  # "easy" | "normal" | "hard"
    spec: Mapping[str, Any]


# Order: easy (weakest first) → normal → hard (strongest last). Matches wide-separation TOML rows.
CPU_VARIANTS: Final[tuple[CpuVariantDef, ...]] = (
    CpuVariantDef(
        "arbalist_emphasis",
        "easy",
        {
            "label": "arbalist_emphasis",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_arbalist": 2.5,
        },
    ),
    CpuVariantDef(
        "arb_focus",
        "easy",
        {
            "label": "arb_focus",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "group_arbalist": 2.5,
        },
    ),
    CpuVariantDef(
        "mage_focus",
        "easy",
        {
            "label": "mage_focus",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "group_mage": 2.5,
            "deploy_center": 0.6,
        },
    ),
    CpuVariantDef(
        "default_like",
        "normal",
        {
            "label": "default_like",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
        },
    ),
    CpuVariantDef(
        "vp_heavy",
        "normal",
        {
            "label": "vp_heavy",
            "vp_scale": 17500.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
        },
    ),
    CpuVariantDef(
        "damage_heavy",
        "normal",
        {
            "label": "damage_heavy",
            "vp_scale": 10000.0,
            "damage_scale": 2.5,
            "win_bonus": 10000000.0,
        },
    ),
    CpuVariantDef(
        "white_mage_emphasis",
        "normal",
        {
            "label": "white_mage_emphasis",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_white_mage": 2.5,
        },
    ),
    CpuVariantDef(
        "knight_emphasis",
        "normal",
        {
            "label": "knight_emphasis",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_knight": 2.5,
        },
    ),
    CpuVariantDef(
        "spell_reach",
        "normal",
        {
            "label": "spell_reach",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_class": [1.0, 1.0, 1.75, 1.75, 1.75],
            "deploy_forward": 0.45,
        },
    ),
    CpuVariantDef(
        "black_mage_emphasis",
        "hard",
        {
            "label": "black_mage_emphasis",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_black_mage": 2.5,
        },
    ),
    CpuVariantDef(
        "barbarian_emphasis",
        "hard",
        {
            "label": "barbarian_emphasis",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "w_barbarian": 2.5,
        },
    ),
    CpuVariantDef(
        "melee_aggressive",
        "hard",
        {
            "label": "melee_aggressive",
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 10000000.0,
            "group_melee": 2.5,
            "deploy_forward": 0.75,
        },
    ),
)


def default_cpu_variant_index() -> int:
    """Index of ``default_like`` (balanced baseline)."""
    for i, v in enumerate(CPU_VARIANTS):
        if v.key == "default_like":
            return i
    return 0


def policy_for_variant_index(index: int) -> ParameterizedHeuristicPolicy:
    v = CPU_VARIANTS[index % len(CPU_VARIANTS)]
    return ParameterizedHeuristicPolicy(heuristic_weights_from_spec(v.spec))
