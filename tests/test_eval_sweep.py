"""eval_sweep: config parse, CSV/report output (tiny run)."""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

import pytest

from motley_crews_play.eval_sweep import (
    SweepConfigError,
    load_variants,
    run_from_toml,
)


def test_load_variants() -> None:
    data = tomllib.loads(
        textwrap.dedent(
            """
            [[variants]]
            label = "a"
            vp_scale = 1.0
            damage_scale = 2.0
            win_bonus = 3.0
            """
        )
    )
    v = load_variants(data)
    assert len(v) == 1
    assert v[0]["label"] == "a"


def test_load_variants_requires_label() -> None:
    data = tomllib.loads(
        """
        [[variants]]
        vp_scale = 1.0
        damage_scale = 1.0
        win_bonus = 1.0
        """
    )
    with pytest.raises(SweepConfigError, match="label"):
        load_variants(data)


def test_run_from_toml_writes_files(tmp_path: Path) -> None:
    csv_p = tmp_path / "out.csv"
    report_p = tmp_path / "out.md"
    data = tomllib.loads(
        textwrap.dedent(
            f"""
            [run]
            seed_start = 0
            seed_count = 1
            max_plies = 40
            calibration_seed_count = 0

            [parallel]
            workers = 1

            [output]
            csv_path = "{csv_p.as_posix()}"
            report_path = "{report_p.as_posix()}"
            summary_top_n = 3
            time_estimate = false

            [[variants]]
            label = "only"
            vp_scale = 10000.0
            damage_scale = 1.0
            win_bonus = 10000000.0
            """
        )
    )
    c_out, r_out = run_from_toml(data)
    assert c_out == csv_p
    assert r_out == report_p
    assert csv_p.is_file()
    assert report_p.is_file()
    text = csv_p.read_text(encoding="utf-8")
    assert "label" in text and "vs_heuristic_class" in text
    md = report_p.read_text(encoding="utf-8")
    assert "Heuristic weight sweep report" in md
    assert "Full table" in md


def test_run_sweep_parallel_two_variants(tmp_path: Path) -> None:
    csv_p = tmp_path / "p.csv"
    report_p = tmp_path / "p.md"
    data = tomllib.loads(
        textwrap.dedent(
            f"""
            [run]
            seed_start = 0
            seed_count = 1
            max_plies = 35
            calibration_seed_count = 0

            [parallel]
            workers = 2

            [output]
            csv_path = "{csv_p.as_posix()}"
            report_path = "{report_p.as_posix()}"
            summary_top_n = 2
            time_estimate = false

            [[variants]]
            label = "a"
            vp_scale = 10000.0
            damage_scale = 1.0
            win_bonus = 10000000.0

            [[variants]]
            label = "b"
            vp_scale = 9000.0
            damage_scale = 1.0
            win_bonus = 10000000.0
            """
        )
    )
    run_from_toml(data)
    assert csv_p.read_text(encoding="utf-8").count("a") >= 1
    assert csv_p.read_text(encoding="utf-8").count("b") >= 1
