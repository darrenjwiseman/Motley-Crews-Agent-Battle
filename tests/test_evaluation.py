"""Batch evaluation, round-robin, Wilson intervals, behavior stats."""

from __future__ import annotations

from motley_crews_play.evaluation import (
    EloTracker,
    MatchBehaviorStats,
    PairwiseRecord,
    action_tuple_has_move,
    action_tuple_has_special,
    behavior_stats_from_log,
    evaluate_pair_swapped,
    evaluate_pair_with_logs,
    perspective_outcome,
    round_robin,
    wilson_score_interval,
)
from motley_crews_play.match import MatchLogger, MatchResult, run_match
from motley_crews_play.policies import (
    ParameterizedHeuristicPolicy,
    RandomPolicy,
    ScriptedHeuristicPolicy,
)


def test_wilson_score_interval_bounds() -> None:
    lo, hi = wilson_score_interval(3, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.5 < hi


def test_wilson_empty() -> None:
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_evaluate_pair_swapped_deterministic() -> None:
    seeds = [0, 1, 2, 3]
    a, b = RandomPolicy(), RandomPolicy()
    r1 = evaluate_pair_swapped(a, b, seeds, max_plies=120)
    r2 = evaluate_pair_swapped(RandomPolicy(), RandomPolicy(), seeds, max_plies=120)
    assert r1.wins == r2.wins
    assert r1.losses == r2.losses
    assert r1.draws == r2.draws
    assert r1.timeouts == r2.timeouts


def test_round_robin_pair_symmetry() -> None:
    rr = round_robin(
        [("x", RandomPolicy()), ("y", RandomPolicy())],
        [0, 1],
        max_plies=80,
    )
    x_vs_y = rr.rows[0].vs["y"]
    y_vs_x = rr.rows[1].vs["x"]
    assert x_vs_y.wins == y_vs_x.losses
    assert x_vs_y.losses == y_vs_x.wins
    assert x_vs_y.draws == y_vs_x.draws
    assert x_vs_y.timeouts == y_vs_x.timeouts


def test_perspective_outcome_timeout() -> None:
    # Force truncation before terminal
    r = run_match(RandomPolicy(), RandomPolicy(), seed=0, max_plies=3)
    mr = MatchResult(final_state=r.final_state, plies=r.plies, stopped_early=True)
    assert perspective_outcome(mr, 0) == "timeout"


def test_elo_tracker_monotonic_win() -> None:
    t = EloTracker(k=32.0, base=1500.0)
    ra, rb = t.update_one(1500.0, 1500.0, 1.0)
    assert ra > 1500.0
    assert rb < 1500.0


def test_elo_from_round_robin_runs() -> None:
    rr = round_robin(
        [
            ("a", RandomPolicy()),
            ("b", RandomPolicy()),
        ],
        [0],
        max_plies=60,
    )
    state = EloTracker().ratings_from_round_robin(rr)
    assert set(state.ratings.keys()) == {"a", "b"}


def test_behavior_stats_from_log_counts() -> None:
    log = MatchLogger()
    log.entries = [
        {"action_tuple": (("m", 0, 0, 0), ("sp", 0, 0, -1, -1, -1, -1))},
        {"action_tuple": (None, ("ba", 0, 1, 1))},
    ]
    s = behavior_stats_from_log(log)
    assert s.plies == 2
    assert s.special_actions == 1
    assert s.turns_with_move == 1


def test_action_tuple_tags() -> None:
    assert action_tuple_has_special((None, ("sp", 0, 0, -1, -1, -1, -1)))
    assert not action_tuple_has_special((None, ("ba", 0, 1, 1)))
    assert action_tuple_has_move((("m", 0, 0, 0), None))


def test_evaluate_pair_with_logs_lengths() -> None:
    rec, stats = evaluate_pair_with_logs(
        RandomPolicy(),
        RandomPolicy(),
        [0],
        max_plies=40,
    )
    assert rec.games == 2
    assert len(stats) == 2
    assert all(isinstance(s, MatchBehaviorStats) for s in stats)


def test_parameterized_heuristic_matches_scripted_defaults() -> None:
    """Same weights → same choices as original heuristic for several plies."""
    rng_a = __import__("random").Random(99)
    rng_b = __import__("random").Random(99)
    from motley_crews_env.engine import initial_play_state, legal_actions, step

    p_script = ScriptedHeuristicPolicy()
    p_param = ParameterizedHeuristicPolicy()
    state_s = initial_play_state()
    state_p = initial_play_state()
    for _ in range(25):
        if state_s.done:
            break
        legal_s = legal_actions(state_s)
        legal_p = legal_actions(state_p)
        assert legal_s == legal_p
        a_s = p_script.choose(state_s, legal_s, rng_a)
        a_p = p_param.choose(state_p, legal_p, rng_b)
        assert a_s == a_p
        state_s = step(state_s, a_s).state
        state_p = step(state_p, a_p).state


def test_heuristic_beats_random_on_average() -> None:
    """Smoke: greedy should beat random more often than not (paired seeds, modest budget)."""
    rec = evaluate_pair_swapped(
        ScriptedHeuristicPolicy(),
        RandomPolicy(),
        list(range(4)),
        max_plies=280,
    )
    assert rec.wins + rec.losses >= 4
    assert rec.wins > rec.losses
