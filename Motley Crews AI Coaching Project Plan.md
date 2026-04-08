# Motley Crews AI Coaching Project Plan

## 1\. Goals & Constraints

- Build a **2‑player AI vs. AI system** that plays *Motley Crews*, a grid‑based tactical fantasy tabletop game where each player controls five figures on an 8×8 board. Each turn you may make one move and one action (attack or special action), and you win by reaching 4 points or eliminating all opposing figures.  
- Allow **human “coaches”** to guide agents via a **small, fixed set of behavior parameters**, not direct move-by-move control.  
- Explicitly **limit both human input and agent learning speed** so the meta does not become immediately hyper‑competitive; favor exploration, stylistic diversity, and readability of play over pure Elo maximization.  
- Target: a system that supports **experiments with multi‑agent RL \+ human preference/parameter feedback**, not production deployment.

---

## 2\. Game Overview for Modeling

Key game facts to capture in the environment:

- Board: **8×8 grid**, with start zones for each player and configurable terrain squares that block movement and actions.  
- Units: Each side has exactly **five classes** — Knight, Barbarian, White Mage, Black Mage, Arbalist — each with movement, attack, reach, life, abilities, and special actions (spells, charge, bomb, etc.).  
- Turn economy: At most **one move and one action per turn**, possibly from different figures; no movement after an action.  
- Win conditions: Gain points for enemy deaths, lose points if resurrected, win at **4 points** or when the opponent has no figures left.  
- Optional achievements / alternative challenges (e.g., win without special actions, specific spell combos) can be used later for multi‑objective rewards.

(If image URLs are available from the rules zine, embed them here as visual references to the core rules layout and unit cards.)

---

## 3\. High‑Level System Architecture

1. **Game Engine / Simulator**  
     
   - Deterministic rules engine with:  
     - Legal move generator  
     - State transition function  
     - Win/lose/draw detection and point tracking  
   - Pluggable **match runner** that pits any two policies against each other.

   

2. **Agent Layer**  
     
   - One **policy module per side**, sharing architecture but with independent weights.  
   - Observation encoder that ingests:  
     - Board layout and terrain  
     - Unit classes, HP, statuses (e.g., containment, cooldowns, “once per game” flags)  
     - Turn and score context  
   - Action space aligned with game engine’s move+action decomposition.

   

3. **Coaching Layer**  
     
   - Small, interpretable **parameter vector** per human coach (“style profile”).  
   - Mapping from parameters → modifications to reward, action preferences, or search heuristics.  
   - UI for coaches to:  
     - Create / edit a style profile  
     - Assign style to an agent for a block of games (“season” or “training window”).

   

4. **Training & Evaluation Layer**  
     
   - Self‑play and coach‑conditioned play pipelines.  
   - Logging, match storage, and analytics (Elo, win‑rate vs baselines, style diversity, stability).

---

## 4\. Phase 1 – Game Environment & Simulator

**4.1. Rules Formalization**

- Translate the PDF rules into a **formal spec**:  
  - Exact movement rules (orthogonal/diagonal per class, terrain blocking).  
  - Damage, HP, death, resurrection, conversion, containment, bombs, once‑per‑game limits, etc.  
  - Ordering and simultaneity of effects (e.g., self‑damage \+ enemy damage from *Curse*).  
- Decide how to represent:  
  - “Once per game” abilities (booleans on unit state).  
  - Status effects with durations (e.g., *Conjure Containment* for N enemy turns).  
  - Team‑ownership changes (converts).

**4.2. State & Action Representation**

- **State representation**  
    
  - Board as tensor(s): \[8×8×(channels for terrain, unit presence, unit class, team, HP buckets, statuses, cooldown flags, score, turn index, etc.)\].  
  - Global context features: remaining specials, points to win, whose turn, repetition or turn limit (if added to avoid long games).


- **Action space**  
    
  - Structured action: `(figure_id, move_target_square, action_type, action_target_square)` where:  
    - `action_type ∈ {none, basic_attack, special_X}`.  
  - Implementation strategy:  
    - Either flatten to a large discrete action index with masking, or use a 2‑stage policy:  
      1. Pick figure & move.  
      2. Pick action & target.

**4.3. Engine Implementation & Tests**

- Implement the engine in a performant language (Python \+ NumPy/JAX/PyTorch is fine for prototype).  
- Unit tests:  
  - Per‑class movement & attack patterns.  
  - Spell behaviors (convert, heal, animate dead, bomb, charge, etc.).  
  - Edge‑cases (no legal moves, simultaneous deaths, terrain blocking, containment effect expiration).  
- Create a **simple random agent** and verify the game is play‑throughable end to end.

Deliverables:

- `motley_crews_env/` library with tested `GameState`, `step()`, and `legal_actions()` APIs.  
- Example scripts: random vs random, scripted scenario tests.

---

## 5\. Phase 2 – Baseline AI Agents (No Coaching Yet)

Goal: establish **weak‑to‑moderate baseline agents** with **deliberately constrained capability**.

**5.1. Baseline Policies**

- Start with **simple, interpretable baselines**:  
  - Rule‑based agent (e.g., greedy for killable targets, preserve low‑HP high‑value units, simple heuristics for healing and conversion).  
  - Lightweight value‑network or policy‑gradient agent with:  
    - Small network (few layers, limited params).  
    - Short horizon or discounting that discourages deep tactical overfitting.

**5.2. Training Regime & Constraints**

- Use **limited training budget**:  
  - Cap on total self‑play episodes and gradient steps per agent “generation”.  
  - Optionally introduce **noise / entropy regularization** to maintain variety and prevent deterministic min‑max play.  
- Avoid heavy search (e.g., no deep MCTS at first), or cap search depth and nodes to keep agents readable and beatable.

**5.3. Baseline Evaluation**

- Metrics:  
  - Win‑rate among baselines and vs random.  
  - Average game length and variance.  
  - Diversity of openings and unit utilization (e.g., how often each special is used).  
- Select 1–2 **reference baseline agents** to serve as fixed opponents for future experiments.

Deliverables:

- Baseline agent implementations and training scripts.  
- Evaluation report (metrics, qualitative examples of play).

---

## 6\. Phase 3 – Human Coaching Layer

Goal: introduce **bounded human influence** via behavior parameters.

**6.1. Define Behavior Parameter Space (Coach Controls)**

Example parameters (tunable, but keep the set small, e.g., 6–10):

- **Aggression vs. preservation**: weight on taking trades vs preserving units.  
- **Healer priority**: how strongly to value healing vs attacking.  
- **Conversion / resurrection preference**: how often to attempt convert/animate plays vs damage.  
- **Terrain & positional risk**: tolerance for entering contested or bomb‑vulnerable zones.  
- **Objective focus**: prioritize:  
  - “Score now” (kills, direct points),  
  - “Board control” (centralization, space),  
  - “Achievements / style goals” (e.g., win without specials).  
- **Special ability thriftiness**: tendency to save once‑per‑game abilities vs spend early.

These parameters should be **exposed as sliders or discrete presets** rather than raw reward weights.

**6.2. Mapping Parameters to Agent Behavior**

Options (can be mixed):

- **Reward shaping**:  
  - Base reward: win/lose \+ small incentives (e.g., short games, unit preservation).  
  - Style reward components multiplied by coach parameters (e.g., heavy positive reward for heals when `healer_priority` is high).  
- **Action preference bias**:  
  - Add logit bonuses/penalties before softmax for actions that align with parameters (e.g., riskier moves penalized when `risk_aversion` is high).  
- **Curriculum constraints**:  
  - Under some styles, mask or down‑weight specific specials or tactics to force distinctive play patterns.

**6.3. Coach Interaction Model & Limits**

- Each coach creates **one or more “style profiles”**, which are immutable during a **training block**:  
  - Example: 100–500 games with a fixed style vector.  
  - Coaches can only update styles between blocks to avoid tight optimization loops.  
- Hard limits on:  
  - **Number of style edits per real‑world week**.  
  - **Number of concurrent style profiles** per coach.  
- Optional: **coach energy / budget**:  
  - Each param change or new style consumes budget, replenished slowly over time.

Deliverables:

- Formal spec of parameter set \+ mapping.  
- UI/JSON schema for coach profiles.  
- Training code that conditions policies on style vector input.

---

## 7\. Phase 4 – Limiting Learning Speed & Hyper‑Competitiveness

Goal: design the progression so agents **don’t instantly converge on a ruthlessly optimal meta**.

**7.1. Training Caps & Seasons**

- **Seasonal training windows**:  
  - Each “season”: fixed number of training steps and matches.  
  - At season end:  
    - Freeze weights.  
    - Run a round‑robin evaluation.  
    - Optionally promote top agents to a “hall of fame” used as opponents in next season.  
- Global caps:  
  - Max gradient steps per day / week.  
  - Max wall‑clock training time or GPU hours.

**7.2. Matchmaking & Ranking**

- Use **soft Elo** or similar rating but:  
  - Cap rating changes per day.  
  - Enforce **wide matchmaking** (agents often face stylistically diverse opponents, not just strongest).  
- Encourage **variety over dominance**:  
  - Track diversity metrics and treat them as first‑class outcomes (e.g., penalize homogenization in tournament seeding).

**7.3. Multi‑Objective & Style Rewards**

- Incorporate **achievement‑like objectives** (e.g., win without specials, specific spell combos) as optional side rewards so no single exploitative win condition dominates every style.  
- Evaluate agents not just on win‑rate but on:  
  - Style adherence (do they reflect coach parameters?).  
  - Spectator metrics (interestingness, variety of tactics).

Deliverables:

- Documented progression rules.  
- Matchmaking & ranking implementation.  
- Season summary scripts/visualizations.

---

## 8\. Phase 5 – Evaluation, UX, and Iteration

**8.1. Quantitative Evaluation**

- For each season / experiment:  
  - Win‑rate matrices between agents.  
  - Correlation between coach parameters and observed behavior (e.g., aggression vs average damage taken/given).  
  - Stability of meta across seasons.

**8.2. Qualitative & UX Evaluation**

- Human review of game logs / replays:  
  - Are the games readable and fun?  
  - Do different coaches clearly produce different play styles?  
- Gather coach feedback:  
  - Is the parameter set understandable?  
  - Are the limits on control and training clear and fair?

**8.3. Iteration Loop**

- Based on evaluation:  
  - Adjust parameter definitions (but keep count small).  
  - Tweak reward shaping and action biases.  
  - Calibrate training caps and season lengths to maintain a “slow meta”.

Deliverables:

- Evaluation dashboards / notebooks.  
- Design doc with findings and proposed next changes.

---

## 9\. Risks & Open Design Questions

- **Degenerate tactics**: Need explicit tests or reward shaping to discourage unfun but legal strategies (e.g., endless stalling, intentional mutual destruction loops).  
- **Complex unit interactions**: Spells like convert, animate dead, bombs, and containment can be tricky to model; thorough testing is critical.  
- **Coach parameter interpretability**: If mapping from parameters to behavior is too opaque, coaching will feel arbitrary.  
- **Meta freeze vs progress**: Need to balance limits on learning with enough progress to keep experimentation interesting.

---

## 10\. Suggested Initial Milestone Breakdown

1. **M1 (Rules → Engine)**  
     
   - Formal rules spec.  
   - Fully tested Motley Crews simulator and random agent.

   

2. **M2 (Baselines)**  
     
   - Rule‑based and small NN baselines.  
   - Evaluation vs random and scripted scenarios.

   

3. **M3 (Coaching v1)**  
     
   - Parameter set \+ mapping to reward/bias.  
   - Coach profile storage and conditioning in training.

   

4. **M4 (Progression System)**  
     
   - Seasons, training caps, matchmaking, basic dashboards.

   

5. **M5 (Pilot Study)**  
     
   - Small number of human coaches.  
   - Analyze style adherence and meta dynamics; refine constraints.

*Written with Glean Assistant*  
