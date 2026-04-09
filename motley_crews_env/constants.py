"""Numeric constants aligned with rules/rules_spec.yaml enums."""

from __future__ import annotations

# Board (rules_spec board.width / height)
BOARD_SIZE: int = 8
BOARD_CELLS: int = BOARD_SIZE * BOARD_SIZE

# enums.class_id order in rules_spec.yaml
CLASS_IDS: tuple[str, ...] = (
    "knight",
    "barbarian",
    "white_mage",
    "black_mage",
    "arbalist",
)
NUM_CLASSES: int = len(CLASS_IDS)
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_IDS)}

# Five figures per side in standard play
FIGURES_PER_SIDE: int = 5
FIGURE_SLOT_MIN: int = 0
FIGURE_SLOT_MAX: int = FIGURES_PER_SIDE - 1

# enums.team_id
TEAM_PLAYER_A: int = 0
TEAM_PLAYER_B: int = 1

# enums.terrain_kind order: open, blocked, water
TERRAIN_OPEN: int = 0
TERRAIN_BLOCKED: int = 1
TERRAIN_WATER: int = 2
NUM_TERRAIN_CHANNELS: int = 3

# Turn sub-phases (representation.global_state.phase)
PHASE_MOVE: int = 0
PHASE_ACTION: int = 1
NUM_PHASES: int = 2

# Scoring (rules_spec scoring.win_points_threshold)
DEFAULT_POINTS_TO_WIN: int = 4

# Starting max HP per class_id index (knight, barbarian, white_mage, black_mage, arbalist)
STARTING_HP_BY_CLASS: tuple[int, ...] = (7, 6, 4, 4, 5)

# Board edges: A bottom, B top (row 0 = north in matrix coords)
PLAYER_A_HOME_ROW: int = 7
PLAYER_B_HOME_ROW: int = 0
DEFAULT_START_COLS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Staging (pre-placement): unit not on board yet
STAGING_COORD: int = -1

# Deployment during setup: each side uses the two rows nearest that edge; rows 2–5 are off-limits
# (center rows 3–4 are the narrow “no man’s land” band; 2 and 5 are still not home rows)
DEPLOY_ROWS_PLAYER_A: tuple[int, ...] = (6, 7)
DEPLOY_ROWS_PLAYER_B: tuple[int, ...] = (0, 1)

# Black Mage Curse: X is chosen by player; engine validates. Upper bound for mask sizing / agents.
MAX_CURSE_X: int = 16

# specials across all classes (rules_spec classes.*.specials), stable index for flat policies
SPECIAL_IDS: tuple[str, ...] = (
    "charge",
    "convert",
    "heal",
    "conjure_containment",
    "curse",
    "magic_bomb",
    "animate_dead",
    "long_eye",
)
NUM_SPECIAL_IDS: int = len(SPECIAL_IDS)
SPECIAL_TO_INDEX: dict[str, int] = {s: i for i, s in enumerate(SPECIAL_IDS)}

# RL / masking: rough upper bounds for composite action enumeration (4.3 refines legality)
MAX_STRUCTURED_TURN_ACTIONS_ESTIMATE: int = (
    (FIGURES_PER_SIDE * (BOARD_CELLS + 1)) * (FIGURES_PER_SIDE * (BOARD_CELLS * 8 + 64))
)
"""Loose upper bound for documentation only; 4.3 should use actual legal counts."""
