"""eval_cli config loading and a tiny end-to-end run."""

from __future__ import annotations

import io
import textwrap
import tomllib

import pytest

from motley_crews_play.eval_cli import EvalConfigError, load_policies, load_run_section, run_from_toml


def test_load_policies_from_minimal_toml() -> None:
    raw = """
    [[policies]]
    name = "a"
    type = "random"

    [[policies]]
    name = "b"
    type = "parameterized_heuristic"
    vp_scale = 5000.0
    damage_scale = 2.0
    win_bonus = 9000000.0
    """
    data = tomllib.loads(textwrap.dedent(raw))
    pols = load_policies(data)
    assert set(pols.keys()) == {"a", "b"}


def test_load_run_section() -> None:
    data = tomllib.loads(
        """
        [run]
        seed_start = 10
        seed_count = 3
        max_plies = 100
        """
    )
    r = load_run_section(data)
    assert list(r.seeds) == [10, 11, 12]
    assert r.max_plies == 100


def test_duplicate_policy_name_rejected() -> None:
    data = tomllib.loads(
        """
        [[policies]]
        name = "x"
        type = "random"
        [[policies]]
        name = "x"
        type = "random"
        """
    )
    with pytest.raises(EvalConfigError, match="Duplicate"):
        load_policies(data)


def test_run_from_toml_tiny_pairwise() -> None:
    data = tomllib.loads(
        textwrap.dedent(
            """
            [run]
            seed_start = 0
            seed_count = 1
            max_plies = 60

            [mode]
            kind = "pairwise"

            [pairwise]
            focus = "h"
            opponent = "r"

            [[policies]]
            name = "r"
            type = "random"

            [[policies]]
            name = "h"
            type = "scripted_heuristic"

            [output]
            print_wilson = false
            elo = false
            behavior = false
            """
        )
    )
    buf = io.StringIO()
    run_from_toml(data, out_stream=buf)
    out = buf.getvalue()
    assert "Pairwise" in out
    assert "h vs r" in out
