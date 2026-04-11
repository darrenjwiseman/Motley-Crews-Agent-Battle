"""eval_sweep: config parse, CSV/report output (tiny run)."""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from motley_crews_play.eval_sweep import (
    BatchSweepAborted,
    SweepConfigError,
    load_variants,
    run_from_toml,
    run_sweep_batch,
    variant_results_from_csv,
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
    assert "How to interpret this report" in md
    assert "Paired sides" in md


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


def test_run_sweep_batch_matches_single_run(tmp_path: Path) -> None:
    csv_single = tmp_path / "single.csv"
    report_single = tmp_path / "single.md"
    csv_batch = tmp_path / "batch.csv"
    report_batch = tmp_path / "batch.md"
    body = """
            [run]
            seed_start = 0
            seed_count = 4
            max_plies = 35
            calibration_seed_count = 0

            [parallel]
            workers = 1

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
    data_single = tomllib.loads(
        textwrap.dedent(
            f"""
            {body}
            [output]
            csv_path = "{csv_single.as_posix()}"
            report_path = "{report_single.as_posix()}"
            summary_top_n = 2
            time_estimate = false
            """
        )
    )
    data_batch = tomllib.loads(
        textwrap.dedent(
            f"""
            {body}
            [output]
            csv_path = "{csv_batch.as_posix()}"
            report_path = "{report_batch.as_posix()}"
            summary_top_n = 2
            time_estimate = false
            """
        )
    )
    run_from_toml(data_single)
    run_sweep_batch(data_batch, batch_shards=4, batch_yes=True)
    master = tmp_path / "batch_master.csv"
    master_md = tmp_path / "batch_master.md"
    assert master.is_file()
    assert master_md.is_file()
    assert "Aggregated from 4 batch shard" in master_md.read_text(encoding="utf-8")
    for i in range(1, 5):
        assert (tmp_path / f"batch_{i}.csv").is_file()
        assert (tmp_path / f"batch_{i}.md").is_file()
    single_rows = variant_results_from_csv(csv_single)
    master_rows = variant_results_from_csv(master)
    assert len(single_rows) == len(master_rows) == 2
    by_label = {r.label: r for r in master_rows}
    for sr in single_rows:
        mr = by_label[sr.label]
        assert mr.vs_heuristic.wins == sr.vs_heuristic.wins
        assert mr.vs_heuristic.losses == sr.vs_heuristic.losses
        assert mr.vs_heuristic.draws == sr.vs_heuristic.draws
        assert mr.vs_heuristic.timeouts == sr.vs_heuristic.timeouts
        assert mr.vs_random.wins == sr.vs_random.wins
        assert mr.vs_random.losses == sr.vs_random.losses
        assert mr.vs_random.draws == sr.vs_random.draws
        assert mr.vs_random.timeouts == sr.vs_random.timeouts


def test_run_sweep_batch_aborts_when_user_declines(tmp_path: Path) -> None:
    csv_p = tmp_path / "out.csv"
    report_p = tmp_path / "out.md"
    data = tomllib.loads(
        textwrap.dedent(
            f"""
            [run]
            seed_start = 0
            seed_count = 2
            max_plies = 30
            calibration_seed_count = 0

            [parallel]
            workers = 1

            [output]
            csv_path = "{csv_p.as_posix()}"
            report_path = "{report_p.as_posix()}"
            summary_top_n = 2
            time_estimate = false

            [[variants]]
            label = "only"
            vp_scale = 10000.0
            damage_scale = 1.0
            win_bonus = 10000000.0
            """
        )
    )
    with patch("motley_crews_play.eval_sweep.sys.stdin.isatty", return_value=True):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(BatchSweepAborted):
                run_sweep_batch(data, batch_shards=2, batch_yes=False)


def test_run_sweep_batch_writes_per_shard_logs(tmp_path: Path) -> None:
    csv_p = tmp_path / "out.csv"
    report_p = tmp_path / "out.md"
    log_dir = tmp_path / "my_logs"
    data = tomllib.loads(
        textwrap.dedent(
            f"""
            [run]
            seed_start = 0
            seed_count = 2
            max_plies = 30
            calibration_seed_count = 0

            [parallel]
            workers = 1

            [output]
            csv_path = "{csv_p.as_posix()}"
            report_path = "{report_p.as_posix()}"
            summary_top_n = 2
            time_estimate = false

            [[variants]]
            label = "only"
            vp_scale = 10000.0
            damage_scale = 1.0
            win_bonus = 10000000.0
            """
        )
    )
    run_sweep_batch(
        data,
        batch_shards=2,
        batch_log_dir=log_dir,
        batch_terminals=False,
        batch_yes=True,
    )
    assert log_dir.is_dir()
    f1 = log_dir / "shard_1.log"
    f2 = log_dir / "shard_2.log"
    assert f1.is_file() and f2.is_file()
    assert "Full evaluation:" in f1.read_text(encoding="utf-8")
    assert "Full evaluation:" in f2.read_text(encoding="utf-8")
