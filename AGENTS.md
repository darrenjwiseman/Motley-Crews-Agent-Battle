# Agent / contributor orientation

Persistent context for this repo lives in:

- **[`.cursor/rules/`](.cursor/rules/)** — Cursor rules (repo map, `motley_crews_env` vs `motley_crews_play` conventions).
- **[`HANDOFF.md`](HANDOFF.md)** — short-lived session notes; read this first when continuing multi-session work.

High-level narrative, **current phase**, and **near-term next steps** are in [`README.md`](README.md) (long-term plan plus sections 5–6), not only the original architecture outline.

## Git workflow (continuity)

- Prefer **small commits** with messages that state intent (what changed and why), so `git log` / `git diff` substitute for re-explaining context.
- For **unfinished work**, use a **WIP branch** and note the branch name plus the next 1–2 steps in `HANDOFF.md`.
- Before ending a session, update `HANDOFF.md` **Current goal**, **Next steps**, and **Files in progress** so the next chat can @-mention that file and proceed.
