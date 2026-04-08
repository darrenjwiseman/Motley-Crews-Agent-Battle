"""
Tensor encoding for observations.

Spatial layout: shape (H, W, C) with H=W=BOARD_SIZE, last dimension channels in CHANNEL_ORDER.

Global vector: fixed-length 1-D array concatenated for policy networks that want a single vector;
use ``encode_observation`` for the full dict with keys ``spatial`` and ``global``.

**RL consumption (4.3+):**

1. **Flat discrete + mask:** 4.3 enumerates legal ``TurnAction`` values and exposes a boolean mask
   over a fixed or dynamic index space; composite size is bounded by per-position legality.
2. **Two-stage policy:** sample ``move`` (actor_slot × destination [+ pass]) from legal moves,
   advance or simulate to post-move state, then sample ``action`` from legal actions — matches
   factored move/action structure in ``types.TurnAction``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from motley_crews_env.constants import (
    BOARD_SIZE,
    DEFAULT_POINTS_TO_WIN,
    NUM_CLASSES,
    NUM_TERRAIN_CHANNELS,
    TERRAIN_OPEN,
)
from motley_crews_env.types import StructuredObservation, TeamId

# Channel indices for spatial tensor (H, W, C), C = SPATIAL_CHANNELS
# 0..2: terrain one-hot (open, blocked, water)
# 3: occupancy
# 4..5: team one-hot (player_a, player_b) — mutually exclusive per occupied cell
# 6..10: class one-hot (5 classes)
# 11: hp normalized [0,1]
# 12: containment
# 13..14: reserved status planes
SPATIAL_CHANNELS: int = NUM_TERRAIN_CHANNELS + 1 + 2 + NUM_CLASSES + 1 + 1 + 2

CHANNEL_TERRAIN_START: int = 0
CHANNEL_TERRAIN_END: int = NUM_TERRAIN_CHANNELS
CHANNEL_OCCUPANCY: int = 3
CHANNEL_TEAM_START: int = 4
CHANNEL_TEAM_END: int = 6
CHANNEL_CLASS_START: int = 6
CHANNEL_CLASS_END: int = 6 + NUM_CLASSES
CHANNEL_HP: int = CHANNEL_CLASS_END
CHANNEL_CONTAINMENT: int = CHANNEL_HP + 1
CHANNEL_RESERVED_A: int = CHANNEL_CONTAINMENT + 1
CHANNEL_RESERVED_B: int = CHANNEL_RESERVED_A + 1

# Global: score_a, score_b (norm), turn_index (norm), current_player one-hot (2), phase one-hot (2),
# points_to_win (norm), repetition_placeholder, turn_limit_placeholder, two extra reserved
GLOBAL_DIM: int = 2 + 1 + 2 + 2 + 1 + 1 + 1 + 2


@dataclass(frozen=True, slots=True)
class ObservationTensorSpec:
    """Fixed shapes and dtypes for tests and engine integration."""

    spatial_shape: tuple[int, int, int] = (BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS)
    spatial_dtype: np.dtype = np.dtype(np.float32)
    global_dim: int = GLOBAL_DIM
    global_dtype: np.dtype = np.dtype(np.float32)


SPEC = ObservationTensorSpec()


def tensor_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "spatial": SPEC.spatial_shape,
        "global": (SPEC.global_dim,),
    }


def structured_observation_to_tensor(
    obs: StructuredObservation,
    *,
    score_normalize_by: float | None = None,
    turn_index_normalize_by: float = 100.0,
) -> NDArray[np.float32]:
    """
    Encode ``StructuredObservation`` into a single float32 tensor of shape (H, W, C).

    If ``score_normalize_by`` is None, uses ``max(obs.points_to_win, DEFAULT_POINTS_TO_WIN)``.
    """
    h, w = BOARD_SIZE, BOARD_SIZE
    out = np.zeros((h, w, SPATIAL_CHANNELS), dtype=np.float32)
    terrain = np.asarray(obs.terrain)
    for y in range(h):
        for x in range(w):
            t = int(terrain[y, x])
            if 0 <= t < NUM_TERRAIN_CHANNELS:
                out[y, x, CHANNEL_TERRAIN_START + t] = 1.0
            else:
                out[y, x, TERRAIN_OPEN] = 1.0
    occ = np.asarray(obs.occupancy)
    out[:, :, CHANNEL_OCCUPANCY] = np.clip(occ.astype(np.float32), 0.0, 1.0)
    team = np.asarray(obs.team)
    for y in range(h):
        for x in range(w):
            tid = int(team[y, x])
            if tid == int(TeamId.PLAYER_A):
                out[y, x, CHANNEL_TEAM_START] = 1.0
            elif tid == int(TeamId.PLAYER_B):
                out[y, x, CHANNEL_TEAM_START + 1] = 1.0
    uclass = np.asarray(obs.unit_class)
    for y in range(h):
        for x in range(w):
            cid = int(uclass[y, x])
            if 0 <= cid < NUM_CLASSES:
                out[y, x, CHANNEL_CLASS_START + cid] = 1.0
    out[:, :, CHANNEL_HP] = np.asarray(obs.hp_normalized, dtype=np.float32)
    out[:, :, CHANNEL_CONTAINMENT] = np.asarray(obs.containment, dtype=np.float32)
    out[:, :, CHANNEL_RESERVED_A] = np.asarray(obs.reserved_status_a, dtype=np.float32)
    out[:, :, CHANNEL_RESERVED_B] = np.asarray(obs.reserved_status_b, dtype=np.float32)
    return out


def structured_observation_to_global_vector(
    obs: StructuredObservation,
    *,
    score_normalize_by: float | None = None,
    turn_index_normalize_by: float = 100.0,
) -> NDArray[np.float32]:
    norm = score_normalize_by
    if norm is None:
        norm = float(max(obs.points_to_win, DEFAULT_POINTS_TO_WIN))
    if norm <= 0:
        norm = 1.0
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0] = float(obs.score_player_a) / norm
    g[1] = float(obs.score_player_b) / norm
    g[2] = float(obs.turn_index) / float(turn_index_normalize_by)
    cp = int(obs.current_player)
    if cp == int(TeamId.PLAYER_A):
        g[3] = 1.0
    elif cp == int(TeamId.PLAYER_B):
        g[4] = 1.0
    ph = int(obs.phase)
    if ph == 0:
        g[5] = 1.0
    elif ph == 1:
        g[6] = 1.0
    g[7] = float(obs.points_to_win) / norm
    g[8] = float(obs.repetition_placeholder)
    g[9] = float(obs.turn_limit_placeholder)
    g[10] = 0.0
    g[11] = 0.0
    return g


def encode_observation(
    obs: StructuredObservation,
    *,
    score_normalize_by: float | None = None,
    turn_index_normalize_by: float = 100.0,
) -> dict[str, NDArray[np.float32]]:
    return {
        "spatial": structured_observation_to_tensor(
            obs,
            score_normalize_by=score_normalize_by,
            turn_index_normalize_by=turn_index_normalize_by,
        ),
        "global": structured_observation_to_global_vector(
            obs,
            score_normalize_by=score_normalize_by,
            turn_index_normalize_by=turn_index_normalize_by,
        ),
    }


def global_vector_layout() -> dict[str, slice]:
    """Slice indices for ``structured_observation_to_global_vector`` output."""
    return {
        "score_a": slice(0, 1),
        "score_b": slice(1, 2),
        "turn_index": slice(2, 3),
        "current_player": slice(3, 5),
        "phase": slice(5, 7),
        "points_to_win": slice(7, 8),
        "repetition_placeholder": slice(8, 9),
        "turn_limit_placeholder": slice(9, 10),
        "reserved_tail": slice(10, GLOBAL_DIM),
    }
