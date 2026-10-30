"""EN ISO 52016-1:2017 6.5.14: moisture content and (de)humidification / latent load."""

import numpy as np
import pytest

from pybuildingenergy.source.utils import (
    DEFAULT_RH_SET_DEHUMIDIFICATION_PCT,
    DEFAULT_RH_SET_HUMIDIFICATION_PCT,
    ISO52016_CA_J_KG_K,
    ISO52016_HWE_J_KG,
    ISO52016_RHO_A_KG_M3,
    _x_set_point,
    _zone_moisture_balance,
)


def _balance(n=1, h_ve=50.0, t_i=20.0, t_e=0.0, rh_e=80.0, vol=0.0, dt=3600.0, hu=30.0, dhu=60.0, **kw):
    f = lambda v: np.full(n, v, dtype=float)
    return _zone_moisture_balance(f(h_ve), f(t_i), f(t_e), f(rh_e), vol, dt, hu, dhu, **kw)


def test_set_point_eq_75_to_77():
    p_sat = 611.2 * np.exp(17.62 * 20.0 / (243.12 + 20.0))
    p_v = 0.5 * p_sat
    assert _x_set_point(20.0, 50.0) == pytest.approx(0.622 * p_v / (101325.0 - p_v))
    assert _x_set_point(20.0, 50.0) == pytest.approx(0.00735, rel=2e-2)  # ~7.3 g/kg


def test_steady_dehumidification_load_eq_72_74():
    # no storage, no internal moisture: Phi = hwe * (H/ca) * (x_ext - x_set_max)
    mb = _balance(t_i=26.0, t_e=32.0, rh_e=60.0, h_ve=50.0, hu=30.0, dhu=50.0)
    expected = ISO52016_HWE_J_KG * 50.0 / ISO52016_CA_J_KG_K * (mb["x_ext"][0] - mb["x_set_max"][0])
    assert mb["phi_dhu_w"][0] == pytest.approx(expected)
    assert mb["phi_hu_w"][0] == 0.0
    assert mb["x_int"][0] == pytest.approx(mb["x_set_max"][0])  # held at the maximum set-point


def test_steady_humidification_load_eq_71_73():
    mb = _balance(t_i=20.0, t_e=0.0, rh_e=80.0, h_ve=50.0, hu=30.0, dhu=60.0)
    expected = ISO52016_HWE_J_KG * 50.0 / ISO52016_CA_J_KG_K * (mb["x_set_min"][0] - mb["x_ext"][0])
    assert mb["phi_hu_w"][0] == pytest.approx(expected)
    assert mb["phi_dhu_w"][0] == 0.0


def test_dead_band_between_the_two_set_points_means_no_load():
    # outdoor air with a moisture content between x_set_min and x_set_max: nothing to do
    mb = _balance(t_i=20.0, t_e=20.0, rh_e=45.0, hu=30.0, dhu=60.0)
    assert mb["phi_hu_w"][0] == 0.0 and mb["phi_dhu_w"][0] == 0.0
    assert mb["x_set_min"][0] < mb["x_int"][0] < mb["x_set_max"][0]


def test_two_set_points_give_a_smaller_load_than_one_set_point():
    one = _balance(hu=50.0, dhu=50.0)["phi_hu_w"][0]
    two = _balance(hu=30.0, dhu=60.0)["phi_hu_w"][0]
    assert 0.0 < two < one


def test_internal_moisture_production_adds_to_the_dehumidification_load():
    base = _balance(t_i=20.0, t_e=20.0, rh_e=55.0, dhu=55.0, hu=30.0)["phi_dhu_w"][0]
    g_int = 0.0002  # kg/s
    withg = _balance(t_i=20.0, t_e=20.0, rh_e=55.0, dhu=55.0, hu=30.0, g_int_kg_s=g_int)["phi_dhu_w"][0]
    assert withg - base == pytest.approx(ISO52016_HWE_J_KG * g_int)  # eq. (72): + G_int
    absorbed = _balance(t_i=20.0, t_e=20.0, rh_e=55.0, dhu=55.0, hu=30.0, g_int_kg_s=g_int, g_abs_kg_s=g_int)["phi_dhu_w"][0]
    assert absorbed == pytest.approx(base)  # eq. (72): - G_abs


def test_storage_term_buffers_a_sudden_humid_spell():
    # hour 1 in the dead band (zone drifts there), hour 2 very humid: with an air volume the
    # drier starting zone absorbs part of the moisture, so the load is smaller than the steady one
    args = dict(n=2, t_i=26.0, t_e=26.0, hu=30.0, dhu=60.0)
    f = lambda v: np.array(v, dtype=float)
    def run(vol):
        return _zone_moisture_balance(f([50.0, 50.0]), f([26.0, 26.0]), f([26.0, 26.0]), f([40.0, 95.0]),
                                      vol, 3600.0, 30.0, 60.0)
    steady, buffered = run(0.0), run(100.0)
    assert steady["phi_dhu_w"][0] == 0.0 and buffered["phi_dhu_w"][0] == 0.0
    assert 0.0 < buffered["phi_dhu_w"][1] < steady["phi_dhu_w"][1]
    # the storage term is exactly rho*V/dt * (x_set_max - x_{t-1}) (eq. 72)
    cap = ISO52016_RHO_A_KG_M3 * 100.0 / 3600.0
    delta = cap * (buffered["x_set_max"][1] - buffered["x_int"][0])
    assert steady["phi_dhu_w"][1] - buffered["phi_dhu_w"][1] == pytest.approx(ISO52016_HWE_J_KG * delta)
    # a load pins the zone at the set-point
    assert buffered["x_int"][1] == pytest.approx(buffered["x_set_max"][1])


def test_free_zone_follows_the_outdoor_moisture_with_a_delay_eq_84():
    free = _zone_moisture_balance(
        np.full(3, 50.0), np.full(3, 20.0), np.full(3, 20.0), np.array([40.0, 40.0, 40.0]),
        100.0, 3600.0, 10.0, 90.0)
    assert free["phi_hu_w"].sum() == 0.0 and free["phi_dhu_w"].sum() == 0.0
    rho_v_dt = ISO52016_RHO_A_KG_M3 * 100.0 / 3600.0
    rho_q = 50.0 / ISO52016_CA_J_KG_K
    x_prev = free["x_ext"][0]  # starts at the outdoor moisture content
    assert free["x_int"][0] == pytest.approx((rho_q * free["x_ext"][0] + rho_v_dt * x_prev) / (rho_q + rho_v_dt))


def test_required_supply_moisture_content_eq_78_79():
    hu = _balance(hu=30.0, dhu=60.0)  # winter humidification
    mech = hu["rho_q_kg_s"][0]
    assert hu["x_sup_hu_req"][0] == pytest.approx(hu["x_ext"][0] + hu["g_hu_kg_s"][0] / mech)
    assert hu["x_sup_hu_req"][0] == pytest.approx(hu["x_set_min"][0])  # all the air covers the load
    dh = _balance(t_i=26.0, t_e=32.0, rh_e=60.0, hu=30.0, dhu=50.0)
    assert dh["x_sup_dhu_req"][0] == pytest.approx(dh["x_set_max"][0])
    # supply through a smaller mechanical flow needs a wider moisture swing
    half = _balance(hu=30.0, dhu=60.0, q_mech_rho_kg_s=hu["rho_q_kg_s"][0] / 2.0)
    assert half["x_sup_hu_req"][0] - half["x_ext"][0] == pytest.approx(2.0 * (hu["x_sup_hu_req"][0] - hu["x_ext"][0]))


def test_system_specific_moisture_content_and_loads_eq_80_83():
    n = 3
    x_ss = np.full(n, 0.010)
    mb = _balance(n=n, t_i=20.0, t_e=10.0, rh_e=60.0, hu=10.0, dhu=90.0, x_sup_ss_kg_kg=x_ss)
    mech = mb["rho_q_kg_s"]
    assert mb["g_ss_hu_kg_s"] == pytest.approx(mech * np.maximum(x_ss - mb["x_ext"], 0.0))  # (81)
    assert mb["phi_ss_hu_w"] == pytest.approx(ISO52016_HWE_J_KG * mb["g_ss_hu_kg_s"])  # (82)
    # no volume and all the air supplied at x_ss: the zone moisture content is x_ss
    assert mb["x_int_ss"] == pytest.approx(x_ss)  # (80)
    dry = _balance(n=n, t_i=20.0, t_e=10.0, rh_e=100.0, hu=10.0, dhu=90.0, x_sup_ss_kg_kg=np.full(n, 0.002))
    assert (dry["phi_ss_dhu_w"] > 0.0).all() and (dry["phi_ss_hu_w"] == 0.0).all()  # (83)


def test_inverted_set_points_are_rejected():
    with pytest.raises(ValueError, match="set-point"):
        _balance(hu=70.0, dhu=40.0)


def test_default_set_points_are_the_en_16798_1_category_limits():
    assert (DEFAULT_RH_SET_HUMIDIFICATION_PCT, DEFAULT_RH_SET_DEHUMIDIFICATION_PCT) == (25.0, 60.0)
