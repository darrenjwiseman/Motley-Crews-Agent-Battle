Motley Crews AI Coaching Project Plan
1. Goals & Constraints

    Build a 2‑player AI vs. AI system that plays Motley Crews, a grid‑based tactical fantasy tabletop game where each player controls five figures on an 8×8 board. Each turn you may make one move and one action (attack or special action), and you win by reaching 4 points or eliminating all opposing figures.

   Allow human “coaches” to guide agents via a small, fixed set of behavior parameters, not direct move-by-move control.

   Explicitly limit both human input and agent learning speed so the meta does not become immediately hyper‑competitive; favor exploration, stylistic diversity, and readability of play over pure Elo maximization.

   Target: a system that supports experiments with multi‑agent RL + human preference/parameter feedback, not production deployment.

3. Game Overview for Modeling
   
    Key game facts to capture in the environment:
   
      Board: 8×8 grid, with start zones for each player and configurable terrain squares that block movement and actions.
   
      Units: Each side has exactly five classes — Knight, Barbarian, White Mage, Black Mage, Arbalist — each with movement, attack, reach, life, abilities, and special actions (spells, charge, bomb, etc.).
   
      Turn economy: At most one move and one action per turn, possibly from different figures; no movement after an action.
   
      Win conditions: Gain points for enemy deaths, lose points if resurrected, win at 4 points or when the opponent has no figures left.
   
      Optional achievements / alternative challenges (e.g., win without special actions, specific spell combos) can be used later for multi‑objective rewards.

4. High‑Level System Architecture

    Game Engine / Simulator
        Deterministic rules engine with:
            Legal move generator
            State transition function
            Win/lose/draw detection and point tracking
        Pluggable match runner that pits any two policies against each other.

    Agent Layer
        One policy module per side, sharing architecture but with independent weights.
        Observation encoder that ingests:
            Board layout and terrain
            Unit classes, HP, statuses (e.g., containment, cooldowns, “once per game” flags)
            Turn and score context
        Action space aligned with game engine’s move+action decomposition.

    Coaching Layer
        Small, interpretable parameter vector per human coach (“style profile”).
        Mapping from parameters → modifications to reward, action preferences, or search heuristics.
        UI for coaches to:
            Create / edit a style profile
            Assign style to an agent for a block of games (“season” or “training window”).

    Training & Evaluation Layer
        Self‑play and coach‑conditioned play pipelines.
        Logging, match storage, and analytics (Elo, win‑rate vs baselines, style diversity, stability).

5. Current phase / roadmap

    Implementation has **diverged in order** from the full architecture in section 4 above:

    1. **Human-playable client** — A Pygame UI (`python -m motley_crews_play --ui`) exposes legal moves and actions so people can play through the real rules. Assets (e.g. sprites) live under `motley_crews_play/`. This validates that the game is understandable and fun to operate before investing in learning stacks.

    2. **Engine correctness and debugging** — Harden `motley_crews_env` against `rules/rules_spec.yaml`, maintain automated tests under `tests/`, and fix edge cases found in play or CI.

    3. **Later** — Return to the **agent, coaching, and training** layers in section 4 once the simulator and human client are trusted.

6. Near-term next steps

    Suggested focus while in phases 1–2 above:

    - **Play testing** — Full matches (human vs CPU/heuristic), all classes and specials, terrain and win/loss paths, setup flow; note bugs and spec mismatches.
    - **UX/UI testing** — Clarity of legal moves and selection, error feedback, responsiveness, keyboard/mouse flow, layout at target resolutions.
    - **Regression** — Run `pytest` after substantive engine or UI changes; add or extend tests for bugs discovered in play.
    - **Backlog** — Track known issues in `HANDOFF.md` during active work (or a short dedicated doc if the list grows).
