"""Pure path helpers for board highlights."""

from motley_crews_play.highlight_geometry import (
    charge_path_cells_exposed,
    l_shaped_path_cells,
    orthogonal_straight_segment_exposed,
)


def test_orthogonal_straight_horizontal() -> None:
    assert orthogonal_straight_segment_exposed(2, 1, 2, 4) == [(2, 1), (2, 2), (2, 3), (2, 4)]


def test_orthogonal_straight_vertical() -> None:
    assert orthogonal_straight_segment_exposed(1, 3, 4, 3) == [(1, 3), (2, 3), (3, 3), (4, 3)]


def test_l_shaped() -> None:
    cells = l_shaped_path_cells(0, 0, 2, 3)
    assert (0, 0) in cells and (0, 3) in cells and (2, 3) in cells
    assert len(cells) >= 4


def test_charge_straight() -> None:
    s = charge_path_cells_exposed(2, 2, 2, 6)
    assert s == {(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)}


def test_charge_rejects_diagonal() -> None:
    assert charge_path_cells_exposed(2, 2, 3, 3) == set()
