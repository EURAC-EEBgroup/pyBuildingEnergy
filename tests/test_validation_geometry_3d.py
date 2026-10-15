"""The 3D test-cell view of the validation reports shows what distinguishes each BESTEST case."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from geometry_3d import building_figure, geometry_caption  # noqa: E402

ASHRAE_CASES = (600, 610, 620, 630, 640, 650, 660, 670, 680, 685, 695,
                900, 910, 920, 930, 940, 950, 960, 980, 985, 995)


def _legend(case, free_float=False):
    fig = building_figure(case, heavy=case >= 900, free_float=free_float)
    return {t.name for t in fig.data if getattr(t, "showlegend", False) and t.name}


@pytest.mark.parametrize("case", ASHRAE_CASES)
def test_every_case_builds_a_figure_with_caption(case):
    assert _legend(case)
    assert geometry_caption(case).strip()


@pytest.mark.parametrize("case,feature", [
    (610, "Overhang / fins"), (910, "Overhang / fins"), (630, "Overhang / fins"), (930, "Overhang / fins"),
    (680, "Increased insulation (Tables 7-25 / 7-31)"), (695, "Increased insulation (Tables 7-25 / 7-31)"),
    (980, "Increased insulation (Tables 7-25 / 7-31)"), (995, "Increased insulation (Tables 7-25 / 7-31)"),
    (650, "Vent fan openings (schematic)"), (950, "Vent fan openings (schematic)"),
    (660, "Windows (low-e, argon-filled, Table 7-17)"), (670, "Windows (single pane, Table 7-21)"),
    (960, "Common wall (0.20 m concrete)"),
])
def test_case_specific_features_are_drawn(case, feature):
    assert feature in _legend(case)


@pytest.mark.parametrize("case", (600, 640, 685, 900, 940, 985))
def test_thermostat_only_cases_have_no_extra_geometry(case):
    legend = _legend(case)
    assert not legend & {"Overhang / fins", "Increased insulation (Tables 7-25 / 7-31)",
                         "Vent fan openings (schematic)"}


def test_free_float_is_stated_and_iso_cells_share_600_900():
    assert "free-floating" in building_figure(900, True, free_float=True).layout.title.text
    assert "Free-floating" in geometry_caption(600, free_float=True)
    assert _legend(600) == _legend(640) and _legend(900) == _legend(940)
