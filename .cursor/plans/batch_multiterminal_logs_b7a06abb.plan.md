---
name: Batch multiterminal logs
overview: Per-shard log files with optional macOS Terminal tail windows; richer supervisor output; aggregated wall-clock ETA from per-shard calibration timings (max of scaled shard estimates).
todos:
  - id: cli-flags
    content: Add --batch-terminals and optional --batch-log-dir; wire run_sweep_batch() params
    status: pending
  - id: redirect-logs
    content: Redirect each shard subprocess stdout/stderr to per-shard log files; set unbuffered env for child
    status: pending
  - id: macos-tails
    content: Implement _open_macos_tail_windows with osascript + path escaping; warn/fallback off darwin
    status: pending
  - id: supervisor-msgs
    content: Improve supervisor-only progress lines (phases, shard i/n done, merge)
    status: pending
  - id: batch-eta
    content: Per-shard calibration timing + aggregate ETA on supervisor (max of scaled estimates); respect time_estimate and calibration_seed_count
    status: pending
  - id: scripts-tests
    content: Add run_sweep_batch_terminals.sh (+.command optional); extend tests for log files without osascript
    status: pending
isProject: false
---

# Multi-window batch sweep logs and supervisor progress

## Current behavior

[`run_sweep_batch`](motley_crews_play/eval_sweep.py) runs shards with `subprocess.run` (no stdout/stderr redirect), so all child output interleaves in the supervisor’s stdout. There is no per-shard log file and no OS integration for extra windows.

Non-batch [`run_from_toml`](motley_crews_play/eval_sweep.py) already supports ETA via `[run].calibration_seed_count` and `[output].time_estimate`: it runs `run_sweep` on the first `cal_n` seeds, measures wall time, scales linearly by `seed_count / cal_n`, and prints one estimate. Batch mode currently forces `calibration_seed_count = 0` on shard subprocesses so calibration does not run four times.

## Approach (multiterminal)

1. **Per-shard log files** — each shard subprocess uses `stdout`/`stderr` redirected to `shard_{i}.log`.
2. **macOS Terminal windows** — optional `osascript` to open Terminal.app with `tail -f` per log (absolute paths, escape for AppleScript).
3. **Supervisor window** — the process running `run_sweep_batch` prints phased status and shard completion; it does not stream child stdout.

Double-clicking a `.command` file opens one Terminal (supervisor); `--batch-terminals` adds four more for tails.

## Aggregated time estimate (supervisor)

**Goal:** Show a **wall-clock ETA for the whole batch** on the supervisor, plus optional per-shard numbers, by **timing each shard’s workload** under a short calibration and then **aggregating** correctly for **parallel** execution.

**Model:** Shards run **in parallel**, so total wall time is **not** the sum of per-shard full-run times; it is approximately **`max(full_shard_estimate_1, …, full_shard_estimate_n)`** (longest straggler). Optionally also print **sum** of estimates as “total CPU-time if sequential” for intuition (clearly labeled).

**Inputs (reuse existing config):**

- `[output].time_estimate` — if false, skip ETA entirely (same semantics as non-batch).
- `[run].calibration_seed_count` (`cal_n`) — if 0 or `cal_n >= seed_count` for a shard, skip or adjust (see edge cases).

**Per-shard calibration (timing “test”):**

For each shard `i` with seed chunk length `L_i`:

1. Let `cal_i = min(cal_n, L_i)` (if `L_i == 0`, skip shard; should not happen when `n_shards <= len(seeds)`).
2. Build the same shard TOML as the full run would use, but with `seed_start` / `seed_count` covering only the **first `cal_i` seeds** of that shard’s range (or equivalently `run_sweep(toml, seeds=those_seeds)` in-process).
3. **Measure wall time** `t_cal_i` for that calibration sweep using the **same** effective worker count as the full shard (`per_shard` / same `run_sweep` parallelism as production). Implementation options (pick one in implementation, prefer clarity over micro-optimization):
   - **A)** `time.perf_counter()` around **in-process** `run_sweep(...)` for shard `i` with calibration seeds only (sequential across shards avoids nested process-pool contention during cal).
   - **B)** Timed **`subprocess.run`** per shard with a temp TOML matching (A), run **sequentially** so calibration reflects cold-start and subprocess overhead similar to full batch.

**Scaling to full shard:**

`est_full_i = t_cal_i * (L_i / cal_i)` (linear in seeds, same assumption as existing `run_from_toml` calibration).

**Aggregate display on supervisor:**

- `ETA_batch_wall ≈ max_i(est_full_i)` — primary headline.
- Print a short table or lines: `shard i: cal X.Xs → est full Y.Ys (Z seeds)` for transparency.
- After the batch **finishes**, optionally print **actual** wall time from batch start to merge complete vs predicted (nice for validation).

**Edge cases:**

- `cal_n == 0` or `time_estimate == false`: no ETA block (or one line: “ETA disabled”).
- `cal_i == L_i`: scale factor is 1; estimate equals calibration time for that shard.
- Uneven shard sizes: `max` still correct for parallel wall time.

**Interaction with multiterminal:** ETA runs in the **supervisor process** before opening tail windows / before or after — **recommend:** run calibration **first** (supervisor only), print ETA, then open tail windows and launch full shard subprocesses so the user sees the estimate before long work begins.

## Implementation details (logs + CLI)

**CLI / API**

- `--batch-terminals` (default off), optional `--batch-log-dir`.
- Thread into `run_sweep_batch(..., batch_terminals: bool, batch_log_dir: Path | None)`.

**`run_sweep_batch`**

- Create log dir; redirect each shard `subprocess.run` stdout/stderr to `shard_i.log`; `PYTHONUNBUFFERED=1` in child env.
- Darwin + `--batch-terminals`: `osascript` to `tail -f` each log; escape paths; tolerate failure.
- Supervisor messages: phases, shard completion, merge; **plus ETA block** when enabled.

**Scripts**

- Optional `run_sweep_batch_terminals.sh` forwarding `--batch-terminals` (default off on base `run_sweep_batch.sh` for CI/headless).

**Tests**

- Default `batch_terminals=False`; assert log files when redirects enabled.
- ETA: unit-test scaling math with mocked times or tiny `cal_n` / tiny sweep (optional; keep fast).

## Risks / limits

- Terminal.app only for auto `tail`; iTerm2 manual.
- `osascript` may fail over SSH; continue with logs only.
- Sequential in-process calibration may take noticeable time before batch; document tradeoff.
