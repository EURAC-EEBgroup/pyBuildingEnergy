"""EN ISO 52016-3:2023 + A1:2025 - adaptive building envelope elements."""

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source import adaptive_envelope_52016_3 as ae


def _surface(**overrides):
    cfg = {
        "type": "roller_blind",
        "position": "external",
        "control_type": "automated",
        "space_category": "residential",
        "states": {
            "retracted": {"g_value": 0.6, "u_value": 1.4, "tau_v": 0.7},
            "extended": {"g_value": 0.1, "u_value": 1.0, "tau_v": 0.05},
        },
    }
    cfg.update(overrides)
    return {"name": "W1", "adaptive_envelope": cfg}


def _context(surface, hours=48, start="2021-07-01", sun=True, occupancy=None, t_e=30.0):
    index = pd.date_range(start, periods=hours, freq="h")
    day = np.array([7 <= h < 19 for h in index.hour])
    sim_df = pd.DataFrame(
        {
            "T2m": t_e,
            "Gd(h)": np.where(day & sun, 100.0, 0.0),
            "Gb(n)": np.where(day & sun, 400.0, 0.0),
        },
        index=index,
    )
    return ae.AdaptiveEnvelopeContext([surface], sim_df, occupancy=occupancy)


def _fill_history(ctx, theta_op, phi_hc):
    ctx.theta_op[:] = theta_op
    ctx.phi_hc[:] = phi_hc


def test_annex_c_tables_are_complete_and_consistent():
    assert sorted(ae._SCENARIOS) == ["C.2", "C.3", "C.4", "C.5", "C.6", "C.7", "C.8"]
    for table in ae._SCENARIOS.values():
        assert sorted(table) == list(range(1, 24))
        for rule_m, corr_m, ext_m, rule_a, corr_a, ext_a in table.values():
            assert 0.0 <= rule_m <= 1.0 and 0.0 <= rule_a <= 1.0
            assert ext_m in (1, 2, 3) and ext_a in (1, 2, 3)
    # spot checks against the printed tables
    assert ae._SCENARIOS["C.2"][6] == (1.0, -0.40, 1, 1.0, 0.0, 1)
    assert ae._SCENARIOS["C.4"][9] == (1.0, -0.40, 2, 1.0, 0.0, 3)
    assert ae._SCENARIOS["C.8"][2] == (1.0, -0.50, 2, 1.0, 0.0, 2)
    assert ae._SCENARIOS["C.3"][23] == (0.5, 0.0, 1, 0.5, 0.0, 1)


@pytest.mark.parametrize(
    "modes, number",
    [
        (("DA", "HE", "TN", "RH", "GN", "LN"), 1),
        (("DA", "HE", "TV", "RL", "GY", "LL"), 2),
        (("DA", "NE", "TN", "RL", "GY", "LN"), 4),
        (("DA", "NE", "TH", "RL", "GN", "LN"), 5),
        (("DA", "NE", "TH", "RH", "GN", "LN"), 6),
        (("DA", "NE", "TV", "RH", "GN", "LL"), 10),
        (("DA", "CO", "TN", "RH", "GY", "LN"), 12),
        (("DA", "CO", "TV", "RH", "GN", "LN"), 17),
        (("DA", "CO", "TV", "RH", "GN", "LL"), 18),
        (("NI", "HE", "TV", "RL", "GN", "LN"), 19),
        (("NI", "NE", "TH", "RL", "GN", "LN"), 20),
        (("NI", "NE", "TV", "RL", "GN", "LN"), 21),
        (("NI", "CO", "TN", "RL", "GN", "LN"), 22),
        (("NI", "CO", "TV", "RL", "GN", "LN"), 23),
    ],
)
def test_state_identification(modes, number):
    assert ae.identify_state_number(*modes) == number


def test_scenario_table_selection():
    assert ae.scenario_table_id("roller_blind", "external", "residential") == "C.2"
    assert ae.scenario_table_id("chromogenic", "internal", "residential") == "C.5"
    assert ae.scenario_table_id("shutter", "integrated", "residential") == "C.3"
    assert ae.scenario_table_id("venetian_blind", "external", "residential") == "C.4"
    assert ae.scenario_table_id("venetian_blind", "internal", "residential") == "C.6"
    assert ae.scenario_table_id("shutter", "external", "office") == "C.7"
    assert ae.scenario_table_id("venetian_blind", "internal", "office") == "C.8"
    with pytest.raises(ValueError):
        ae.scenario_table_id("shutter", "internal", "residential")


def test_radiation_limits_follow_table_c1():
    to_w = 1000.0 / 115.0
    rl, rh = ae.radiation_limits_w_m2("AUT", "RES", "CO", "TV")
    assert (rl, rh) == pytest.approx((10.0 * to_w, 15.0 * to_w))
    rl, rh = ae.radiation_limits_w_m2("MAN", "OFFHE", "NE", "TH")
    assert (rl, rh) == pytest.approx((37.5 * to_w, 72.5 * to_w))
    assert ae.radiation_limits_w_m2("AUT", "RES", "HE", "TV") is None
    assert ae.radiation_limits_w_m2("AUT", "RES", "NE", "TN") is None


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="type"):
        ae.parse_adaptive_envelope(_surface(type="curtain"))
    with pytest.raises(ValueError, match="control_type"):
        ae.parse_adaptive_envelope(_surface(control_type="telepathy"))
    with pytest.raises(ValueError, match="slats_open"):
        ae.parse_adaptive_envelope(_surface(type="venetian_blind"))
    with pytest.raises(ValueError, match="no manual control limits"):
        ae.parse_adaptive_envelope(
            _surface(control_type="manual", space_category="office", system="heating_cooling")
        )
    bad = _surface()
    bad["adaptive_envelope"]["states"]["extended"]["g_value"] = 1.5
    with pytest.raises(ValueError, match="g_value"):
        ae.parse_adaptive_envelope(bad)


def test_hc_mode_uses_previous_completed_day_from_0700():
    ctx = _context(_surface(), hours=72)
    _fill_history(ctx, 21.5, 0.0)
    assert ctx.hc_mode(0) == "NE"  # no completed previous day yet
    # first full 07h-07h period ends at t = 24 + 7 = 31 (2021-07-02 07:00)
    assert ctx.hc_mode(31) == "NE"
    ctx.phi_hc[7:31] = 500.0  # heating during that day
    ctx._hc_cache.clear()
    assert ctx.hc_mode(31) == "HE" and ctx.hc_mode(54) == "HE"
    assert ctx.hc_mode(30) == "NE"  # still the previous (incomplete) period
    ctx.phi_hc[7:31] = -500.0
    ctx._hc_cache.clear()
    assert ctx.hc_mode(40) == "CO"
    # no load: temperature limits
    ctx.phi_hc[:] = 0.0
    ctx.theta_op[:] = 19.0
    ctx._hc_cache.clear()
    assert ctx.hc_mode(40) == "HE"
    ctx.theta_op[:] = 24.0
    ctx._hc_cache.clear()
    assert ctx.hc_mode(40) == "CO"


def test_hc_mode_outdoor_method_am2():
    ctx = _context(_surface(), hours=72, t_e=10.0)
    assert ctx.hc_mode(40, "AM2") == "HE"
    ctx.theta_e[:] = 14.0
    ctx._hc_cache.clear()
    assert ctx.hc_mode(40, "AM2") == "NE"
    ctx.theta_e[:] = 20.0
    ctx._hc_cache.clear()
    assert ctx.hc_mode(40, "AM2") == "CO"


def test_tint_mode_uses_previous_interval():
    ctx = _context(_surface())
    for value, mode in [(21.0, "TN"), (22.5, "TH"), (24.5, "TV")]:
        ctx.theta_op[9] = value
        assert ctx.tint_mode(10) == mode
    assert ctx.tint_mode(0) == "TN"


def test_automated_residential_closes_in_cooling_with_high_sun():
    surface = _surface()
    ctx = _context(surface)
    _fill_history(ctx, 26.0, -300.0)  # hot, cooling during the previous day
    ctrl = ctx.controllers["W1"]
    st = ctrl.state_at(ctx, 36, 400.0)  # day 2, 12:00, 400 W/m2 >> 15 klux/115
    assert st.modes == "DA-CO-TV-RH-GN-LN"
    assert st.state_number == 17
    assert st.fraction == 1.0
    assert st.properties.g == pytest.approx(0.1)
    # low sun: retracted
    st = ctrl.state_at(ctx, 37, 20.0)
    assert st.modes.endswith("RL-GN-LN") and st.fraction == 0.0
    assert st.properties.g == pytest.approx(0.6)


def test_radiation_hysteresis_keeps_previous_mode_between_limits():
    ctx = _context(_surface())
    _fill_history(ctx, 26.0, -300.0)
    ctrl = ctx.controllers["W1"]
    # CO-TV limits (AUT-RES): RL 10 klux (87 W/m2), RH 15 klux (130 W/m2)
    assert ctrl.state_at(ctx, 34, 200.0).modes.split("-")[3] == "RH"
    assert ctrl.state_at(ctx, 35, 100.0).modes.split("-")[3] == "RH"  # in between
    assert ctrl.state_at(ctx, 36, 50.0).modes.split("-")[3] == "RL"
    assert ctrl.state_at(ctx, 37, 100.0).modes.split("-")[3] == "RL"  # in between


def test_heating_mode_harvests_solar_heat():
    ctx = _context(_surface())
    _fill_history(ctx, 18.0, 800.0)
    st = ctx.controllers["W1"].state_at(ctx, 36, 600.0)
    assert st.modes.startswith("DA-HE-TN") and st.state_number == 1
    assert st.fraction == 0.0


def test_manual_control_applies_non_responding_fraction():
    surface = _surface(control_type="manual")
    ctx = _context(surface, occupancy=np.ones(48))
    _fill_history(ctx, 26.0, -300.0)
    st = ctx.controllers["W1"].state_at(ctx, 36, 600.0)
    # MAN-RES, CO-TV, RH: rule 1.00, 20 % (Table C.2 state 17) does not respond
    assert st.state_number == 17 and st.fraction == pytest.approx(0.8)
    assert st.properties.g == pytest.approx(0.2 * 0.6 + 0.8 * 0.1)
    assert st.properties.u == pytest.approx(0.2 * 1.4 + 0.8 * 1.0)
    # at night nothing is extended by rule, 10 % stay extended (state 23)
    st = ctx.controllers["W1"].state_at(ctx, 44, 0.0)  # 20:00 awake
    assert st.state_number == 23 and st.fraction == pytest.approx(0.1)


def test_manual_control_holds_position_when_occupant_not_awake():
    surface = _surface(control_type="manual", space_category="office", system="heating")
    occupancy = np.zeros(48)
    occupancy[34] = 1.0  # occupied at 10:00 of day 2 only
    ctx = _context(surface, occupancy=occupancy)
    _fill_history(ctx, 26.0, 0.0)
    ctrl = ctx.controllers["W1"]
    st_occupied = ctrl.state_at(ctx, 34, 600.0)
    st_empty = ctrl.state_at(ctx, 35, 0.0)  # unoccupied: previous position kept
    assert st_occupied.fraction == pytest.approx(0.5)  # Table C.7 state 17: 1.00 - 0.50
    assert st_empty.held and st_empty.fraction == st_occupied.fraction


def test_automated_office_ignores_glare_and_correction_when_unoccupied():
    surface = _surface(space_category="office", system="heating")
    occupancy = np.zeros(48)
    occupancy[34] = 1.0
    ctx = _context(surface, occupancy=occupancy)
    _fill_history(ctx, 21.0, 0.0)  # NE, TN
    ctrl = ctx.controllers["W1"]
    # E = 115 * 400 = 46 klux -> Pdglare = 0.25*6.22e-5*46000+0.184 = 0.90 > 0.45
    occupied = ctrl.state_at(ctx, 34, 400.0)
    unocc = ctrl.state_at(ctx, 35, 400.0)
    assert occupied.modes.split("-")[4] == "GY"
    assert unocc.modes.split("-")[4] == "GN"


def test_low_daylight_uses_formula_4_with_tau_of_the_interim_state():
    # interim state 17 is fully extended: tau_V = 0.05
    low = {"a_dayl": 0.1, "b_dayl": 0.0, "H_gl_1m": 1.0}
    ctx = _context(_surface(control={"daylight": low}))
    _fill_history(ctx, 26.0, -300.0)
    # E_desk = 0.1 * 0.05 * 46000 lx = 230 lx < 300 lx -> LL
    assert ctx.controllers["W1"].state_at(ctx, 36, 400.0).modes.endswith("LL")
    high = {"a_dayl": 0.5, "b_dayl": 0.0, "H_gl_1m": 1.0}
    ctx = _context(_surface(control={"daylight": high}))
    _fill_history(ctx, 26.0, -300.0)
    # E_desk = 0.5 * 0.05 * 46000 lx = 1150 lx -> LN
    assert ctx.controllers["W1"].state_at(ctx, 36, 400.0).modes.endswith("LN")


def test_venetian_blind_uses_slat_state_of_the_scenario():
    states = {
        "retracted": {"g_value": 0.6, "u_value": 1.4},
        "extended": {
            "slats_open": {"g_value": 0.5, "u_value": 1.3},
            "slats_half_open": {"g_value": 0.3, "u_value": 1.2},
            "slats_closed": {"g_value": 0.1, "u_value": 1.1},
        },
    }
    surface = _surface(type="venetian_blind", states=states)
    ctx = _context(surface)
    _fill_history(ctx, 26.0, -300.0)
    st = ctx.controllers["W1"].state_at(ctx, 36, 600.0)
    # Table C.4, automated, state 17: rule 1.00, slat state 3 (closed)
    assert st.ext_state == 3 and st.properties.g == pytest.approx(0.1)


def test_night_daytime_detection_uses_calculated_irradiance_or_fixed_hours():
    ctx = _context(_surface())
    assert ctx.is_day(36) and not ctx.is_day(30)
    assert ctx.is_day(30, fixed_hours=(6, 20))  # 06:00 inside fixed window


def test_environmental_chromogenic_follows_formula_1_with_hysteresis():
    surface = _surface(
        type="chromogenic",
        control_type="environmental",
        control={"environment_variable": "solar_irradiance", "hysteresis": 100.0},
        states={
            "state_min": {"x": 100.0, "g_value": 0.6, "u_value": 1.2, "tau_v": 0.7},
            "state_max": {"x": 700.0, "g_value": 0.1, "u_value": 1.2, "tau_v": 0.05},
        },
    )
    ctx = _context(surface)
    ctrl = ctx.controllers["W1"]
    rising = ctrl.state_at(ctx, 10, 400.0)  # (400-100+50)/600
    assert rising.fraction == pytest.approx(350.0 / 600.0)
    assert rising.properties.g == pytest.approx(0.6 + 350.0 / 600.0 * (0.1 - 0.6))
    falling = ctrl.state_at(ctx, 11, 300.0)  # (300-100-50)/600
    assert falling.fraction == pytest.approx(150.0 / 600.0)
    assert ctrl.state_at(ctx, 12, 1000.0).fraction == 1.0  # clipped
    assert ctrl.state_at(ctx, 13, 0.0).fraction == 0.0


def test_properties_are_cached_per_step():
    ctx = _context(_surface())
    _fill_history(ctx, 26.0, -300.0)
    first = ctx.properties(_surface(), 36, 400.0)
    ctx.theta_op[:] = 18.0  # must not change an already decided step
    assert ctx.properties(_surface(), 36, 400.0) == first


def test_state_log_dataframe_columns():
    ctx = _context(_surface())
    _fill_history(ctx, 26.0, -300.0)
    ctx.properties(_surface(), 36, 400.0)
    log = ctx.to_dataframe()
    for suffix in ("modes", "state", "fraction", "ext_state", "g", "u", "tau_v"):
        assert f"adaptive_W1_{suffix}" in log.columns


def test_kpis_against_reference():
    idx = pd.date_range("2021-01-01", periods=4, freq="h")
    reference = pd.DataFrame(
        {"Q_H": [1000.0] * 4, "Q_C": [500.0] * 4, "T_op": [19.0, 26.0, 21.0, 21.0]}, index=idx
    )
    adaptive = pd.DataFrame(
        {"Q_H": [1100.0] * 4, "Q_C": [250.0] * 4, "T_op": [19.0, 24.0, 21.0, 21.0]}, index=idx
    )
    kpi = ae.adaptive_envelope_kpis(adaptive, reference, useful_floor_area_m2=10.0)
    assert kpi["dQ_H_nd_kWh_m2"] == pytest.approx(0.04)
    assert kpi["dQ_C_nd_kWh_m2"] == pytest.approx(-0.1)
    assert kpi["dQ_C_nd_pct"] == pytest.approx(-50.0)
    assert kpi["dT_high_Kh"] == pytest.approx(-1.0)
    assert kpi["dT_low_Kh"] == pytest.approx(0.0)


def test_context_is_never_copied_with_dataframe_attrs():
    import copy

    ctx = _context(_surface())
    assert copy.deepcopy(ctx) is ctx and copy.copy(ctx) is ctx
    ctx.index.to_series().to_frame().attrs["c"] = ctx  # pandas copies attrs on access


# --- additions: 6.7 discrete states, glare policy, ordering, reports ---------------------


def test_chromogenic_states_must_be_ordered():
    bad = _surface(
        type="chromogenic", control_type="environmental",
        states={
            "state_min": {"x": 700.0, "g_value": 0.6, "u_value": 1.2},
            "state_max": {"x": 100.0, "g_value": 0.1, "u_value": 1.2},
        },
    )
    with pytest.raises(ValueError, match="lower than state_max"):
        ae.parse_adaptive_envelope(bad)
    equal = _surface(
        type="chromogenic", control_type="environmental",
        states={
            "state_min": {"x": 100.0, "g_value": 0.6, "u_value": 1.2},
            "state_max": {"x": 100.0, "g_value": 0.1, "u_value": 1.2},
        },
    )
    with pytest.raises(ValueError, match="lower than state_max"):
        ae.parse_adaptive_envelope(equal)


def _discrete_surface(hysteresis=0.0):
    return _surface(
        type="chromogenic", control_type="environmental",
        control={"environment_variable": "solar_irradiance", "hysteresis": hysteresis},
        states={"discrete": [
            {"x": 0.0, "g_value": 0.6, "u_value": 1.2, "tau_v": 0.7},
            {"x": 300.0, "g_value": 0.3, "u_value": 1.2, "tau_v": 0.3},
            {"x": 600.0, "g_value": 0.1, "u_value": 1.2, "tau_v": 0.05},
        ]},
    )


def test_discrete_environmental_states_change_when_the_next_state_is_reached():
    ctx = _context(_discrete_surface())
    ctrl = ctx.controllers["W1"]
    gs = [ctrl.state_at(ctx, t, x).properties.g for t, x in enumerate([100.0, 299.0, 300.0, 599.0, 600.0, 900.0, 50.0])]
    assert gs == [0.6, 0.6, 0.3, 0.3, 0.1, 0.1, 0.6]


def test_discrete_environmental_states_with_hysteresis():
    ctx = _context(_discrete_surface(hysteresis=100.0))
    ctrl = ctx.controllers["W1"]
    # rising: threshold effectively 50 W/m2 lower (Formula 1 convention +dX/2)
    assert ctrl.state_at(ctx, 0, 100.0).properties.g == 0.6
    assert ctrl.state_at(ctx, 1, 260.0).properties.g == 0.3
    assert ctrl.state_at(ctx, 2, 340.0).properties.g == 0.3
    # falling: threshold effectively 50 W/m2 higher (state kept down to 350 W/m2)
    assert ctrl.state_at(ctx, 3, 320.0).properties.g == 0.6
    assert ctrl.state_at(ctx, 4, 200.0).properties.g == 0.6


def test_discrete_states_need_ascending_x():
    bad = _discrete_surface()
    bad["adaptive_envelope"]["states"]["discrete"][1]["x"] = 0.0
    with pytest.raises(ValueError, match="ascending"):
        ae.parse_adaptive_envelope(bad)


def test_glare_policy_is_selectable():
    # Formula (3) limit ~17 106 lx (148.7 W/m2), Table 33 note a: 4 500 lx (39.1 W/m2)
    def glare(**control):
        surface = _surface(space_category="office", system="heating", control=control)
        ctx = _context(surface, occupancy=np.ones(48))
        _fill_history(ctx, 21.0, 0.0)
        return ctx.controllers["W1"].state_at(ctx, 34, 100.0).modes.split("-")[4]

    assert glare() == "GN"  # 11 500 lx < 17 106 lx
    assert glare(glare_policy="table_33_note") == "GY"  # 11 500 lx > 4 500 lx
    assert glare(glare_limit_lx=10000.0) == "GY"
    assert ae.GLARE_LIMIT_LX_FORMULA == pytest.approx(17106.0, abs=1.0)
    with pytest.raises(ValueError, match="glare_policy"):
        ae.parse_adaptive_envelope(_surface(control={"glare_policy": "x"}))


def _hourly_with_log(adaptive_kwh_scale=1.0):
    idx = pd.date_range("2021-01-01", periods=24 * 59, freq="h")  # Jan + Feb
    n = len(idx)
    state = np.where(idx.hour >= 12, 17, 1)
    g = np.where(idx.hour >= 12, 0.1, 0.6)
    df = pd.DataFrame(
        {
            "Q_H": 1000.0 * adaptive_kwh_scale, "Q_C": 100.0, "T_op": 21.0, "T_ext": 5.0,
            "adaptive_W_modes": "x", "adaptive_W_state": state, "adaptive_W_fraction": 0.0,
            "adaptive_W_ext_state": 1, "adaptive_W_g": g, "adaptive_W_u": 1.2,
            "adaptive_W_tau_v": 0.5, "adaptive_W_I_sol": np.where(idx.hour >= 12, 200.0, 0.0),
        },
        index=idx,
    )
    assert len(df) == n
    return df


def test_monthly_report_and_statistics():
    adaptive, reference = _hourly_with_log(1.1), _hourly_with_log(1.0).filter(["Q_H", "Q_C", "T_op", "T_ext"])
    monthly = ae.adaptive_envelope_monthly_report(adaptive, reference, useful_floor_area_m2=100.0)
    assert list(monthly.index) == [1, 2, "year"]
    # +100 W x 744 h in January = 74.4 kWh / 100 m2
    assert monthly.loc[1, "dQ_H_nd_kWh_m2"] == pytest.approx(0.744)
    assert monthly.loc["year", "dQ_H_nd_pct"] == pytest.approx(10.0)
    # g weighted by irradiance: only the hours with sun (g = 0.1) count
    assert monthly.loc["year", "W:g_gl_mn"] == pytest.approx(0.1)
    assert monthly.loc["year", "W:U_w_mn"] == pytest.approx(1.2)
    stats = ae.adaptive_envelope_state_statistics(adaptive)["W"]
    assert stats["occurrences"] == {1: 708, 17: 708}
    assert stats["changes"] == 117  # 59 days to g = 0.1, 58 back to g = 0.6
    assert stats["mean_hours_without_change"] == pytest.approx(len(adaptive) / 117)


def test_conformity_report_lists_options_and_gaps():
    surface = _surface(space_category="office", system="heating_cooling", control={"hc_method": "AM2"})
    surface["name"] = "South window"
    report = ae.adaptive_envelope_conformity_report({"building_surface": [surface]})
    el = report["elements"]["South window"]
    assert el["control_type"] == "CNTRL_AUTO" and el["annex_c_table"] == "C.7"
    assert "AM2" in el["methods"]["HC-MODE"]
    assert "17106" in el["methods"]["GLARE-MODE"] or "17107" in el["methods"]["GLARE-MODE"]
    text = ae.format_conformity_report(report)
    assert "Not implemented" in text and "ventilated cavity" in text
