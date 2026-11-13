"""Check against the EPB Center case study on EN ISO 52016-3 (2023-09-30).

``data/epb_case_soff_ath_exven_aut_heco.csv.gz`` holds the hourly inputs and
results of the sheet AdaptW of the EPB Center spreadsheet (case SOff, Athens,
external venetian blind, automated, heating and cooling; 9 504 h incl. the
December warm-up).  The control logic is fed with the sheet's own hourly inputs
and its states are compared hour by hour.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source import adaptive_envelope_52016_3 as ae

DATA = Path(__file__).resolve().parent / "data" / "epb_case_soff_ath_exven_aut_heco.csv.gz"

# DG-EXVBLD of the spreadsheet library (case study 6.1.2.4)
DG_EXVBLD = {
    "retracted": {"g_value": 0.653, "u_value": 1.063, "tau_v": 0.814},
    "extended": {
        "slats_open": {"g_value": 0.142, "u_value": 0.936, "tau_v": 0.178},
        "slats_half_open": {"g_value": 0.074, "u_value": 0.931, "tau_v": 0.085},
        "slats_closed": {"g_value": 0.03, "u_value": 0.914, "tau_v": 0.02},
    },
}
R_SI_PLUS_R_SE = 1.0 / 0.936 - 1.0 / 1.1156031885274038  # sheet: U <-> 1/Rc;w


@pytest.fixture(scope="module")
def run():
    sheet = pd.read_csv(DATA)
    n = len(sheet)
    surface = {
        "name": "W",
        "adaptive_envelope": {
            "type": "venetian_blind", "position": "external", "control_type": "automated",
            "space_category": "office", "system": "heating_cooling", "states": DG_EXVBLD,
            "control": {"daylight": {"a_dayl": 0.02, "b_dayl": 30.0, "H_gl_1m": 1.4}},
        },
    }
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    sim_df = pd.DataFrame({"T2m": sheet["theta_e_C"].to_numpy()}, index=index)
    occupancy = (sheet["occupancy"] == "OCC-AW").to_numpy(dtype=float)
    ctx = ae.AdaptiveEnvelopeContext([surface], sim_df, occupancy=occupancy)
    ctx.theta_op[:-1] = sheet["theta_op_prev_C"].to_numpy()[1:]  # theta_op[t-1] shown at row t
    ctx._irradiance_day = (sheet["day"] == "DA").to_numpy()
    phi24, op24 = sheet["phi_hc_24_W"].to_numpy(), sheet["theta_op_24_C"].to_numpy()
    ctx.hc_mode = lambda t, method="AM1": (
        "HE" if phi24[t] > 0 or op24[t] < 20 else "CO" if phi24[t] < 0 or op24[t] > 23 else "NE"
    )
    ctrl = ctx.controllers["W"]
    states = [ctrl.state_at(ctx, t, float(i)) for t, i in enumerate(sheet["I_sol_W_m2"])]
    return sheet, ctx, states


def test_state_sequence_matches_the_spreadsheet(run):
    sheet, _, states = run
    mine = np.array([s.state_number for s in states])
    agree = np.mean(mine == sheet["state"].to_numpy())
    assert agree > 0.999  # 3 of 9 504 hours differ: glare limit 17 250 lx (sheet) vs 17 106 lx (Formula 3)


def test_modes_match_the_spreadsheet(run):
    sheet, _, states = run
    modes = [s.modes.split("-") for s in states]
    assert [m[2] for m in modes] == sheet["tint"].tolist()  # TINT-MODE: 100 %
    assert np.mean([m[4] for m in modes] == sheet["glare"].to_numpy()) > 0.999


def test_properties_match_the_spreadsheet_where_the_state_agrees(run):
    sheet, _, states = run
    same = np.array([s.state_number for s in states]) == sheet["state"].to_numpy()
    g = np.array([s.properties.g for s in states])
    tau = np.array([s.properties.tau_v for s in states])
    u = np.array([s.properties.u for s in states])
    u_sheet = 1.0 / (1.0 / sheet["u_over_rc"].to_numpy() + R_SI_PLUS_R_SE)
    np.testing.assert_allclose(g[same], sheet["g"].to_numpy()[same], atol=1e-9)
    np.testing.assert_allclose(tau[same], sheet["tau_v"].to_numpy()[same], atol=1e-9)
    np.testing.assert_allclose(u[same], u_sheet[same], atol=1e-6)


def test_previous_day_window_matches_the_sheet_average(run):
    sheet, ctx, _ = run
    op24 = sheet["theta_op_24_C"].to_numpy()
    for t in range(60, len(sheet), 37):
        offset = (ctx.clock[t] - 7) % 24
        end = t - int(round(offset))
        assert np.mean(ctx.theta_op[end - 24:end]) == pytest.approx(op24[t], abs=1e-6)


def test_case_study_u_values_of_the_states_follow_from_area_weighting():
    # 6.2.2.1.2: residential, double glazing; "U = 1,00 internal vs 1,05 external in
    # state 1 (80 % resp. 10 % extended, slats open); 0,97 vs 0,91 in state 19"
    ret = ae.OpticalState(0.653, 1.063)
    int_open, int_closed = ae.OpticalState(0.478, 0.978), ae.OpticalState(0.294, 0.941)
    ext_open, ext_closed = ae.OpticalState(0.142, 0.936), ae.OpticalState(0.03, 0.914)
    assert ret.mixed(int_open, 0.8).u == pytest.approx(1.00, abs=0.006)
    assert ret.mixed(ext_open, 0.1).u == pytest.approx(1.05, abs=0.005)
    assert ret.mixed(int_closed, 0.8).u == pytest.approx(0.97, abs=0.006)
    assert ret.mixed(ext_closed, 1.0).u == pytest.approx(0.91, abs=0.005)


@pytest.mark.parametrize(
    "table, state, fraction, slats",
    [  # case study 6.2.2.1.1/6.2.2.1.4: manual internal venetian (Table C.6), 20 % do not respond
        ("C.6", 19, 0.8, 3), ("C.6", 23, 0.8, 2), ("C.6", 16, 0.8, 1), ("C.6", 17, 0.8, 3),
        # external venetian (Table C.4): 60 % / slats 45 deg in high-radiation states
        ("C.4", 6, 0.6, 2), ("C.4", 16, 0.6, 1), ("C.4", 17, 0.8, 3),
        # external roller blind (Table C.2): 10 % at night, 60 %, 80 %
        ("C.2", 19, 0.1, 1), ("C.2", 14, 0.6, 1), ("C.2", 17, 0.8, 1),
    ],
)
def test_case_study_manual_states(table, state, fraction, slats):
    rule, corr, ext = ae.scenario_rows(table)[state][:3]
    assert rule + corr == pytest.approx(fraction)
    assert ext == slats


def test_annex_c_editions_differ_only_in_table_c4_night():
    # the case study (and its spreadsheet) use the 2023 text: external venetian blinds
    # with manual/motorized control are fully extended at night; 2026 prints 0,00 + 0,10
    for number in range(19, 24):
        r23 = ae.scenario_rows("C.4", "2023")[number]
        r26 = ae.scenario_rows("C.4", "2026")[number]
        assert (r23[0], r23[1]) == (1.0, 0.0) and (r26[0], r26[1]) == (0.0, 0.1)
        assert r23[2:] == r26[2:]
    for table in ("C.2", "C.3", "C.5", "C.6", "C.7", "C.8"):
        assert ae.scenario_rows(table, "2023") == ae.scenario_rows(table, "2026")
