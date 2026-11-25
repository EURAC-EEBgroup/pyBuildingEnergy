"""Central air handling unit serving several thermal zones (EN 16798-5-1, 6.4.2, 6.4.3.2.1).

One AHU feeds a duct network to several zones.  Each zone has a terminal unit: constant
volume, or VAV between a minimum and a maximum flow.  Per time step:

1. Zone flow requirement.  A VAV terminal sizes its flow from the zone sensible load of
   the previous step: the load the ideal zone system still covered plus the heat the VAV
   supply gave, divided by the supply-to-zone temperature difference, between the
   minimum and maximum flow.  When the supply cannot cover the load (for example cold
   supply and a heating load) the terminal goes to its minimum flow.  A constant volume
   terminal asks for its maximum flow.
2. AHU flow, AIR_FLOW_CTRL (23a)-(23d) and (24a)-(24d): NO_CTRL, ON_OFF, MULTI_STAGE or
   VARIABLE (with the minimum flow ratio f_q;min), applied to the sum of the zone flows.
3. Zone flows (6)-(7): the AHU flow shared in proportion to the zone requirements.
4. Extract air: flow-weighted mix of the zone air temperatures and moisture contents of
   the previous step.
5. One AHU step (:func:`calculate_sensible_ahu_step`) with distribution, leakage,
   recirculation, frost protection, adiabatic cooling, coils and fans.  The supply leaves
   the network at the same temperature for every zone.  When a zone extract exceeds what
   it receives, the difference enters the zone from outdoors.

Configuration (``building_parameters.central_ahus``, list): the AHU keys of a
``mechanical_supply`` component (supply temperature control, heat recovery, coils, fans,
ducts, recirculation, frost, adiabatic cooling, humidity set-points) plus:

* ``name``;
* ``air_flow_control``: ``no_ctrl``, ``on_off``, ``multi_stage`` or ``variable``
  (default ``variable`` when a zone has a VAV terminal, else ``on_off``);
* ``supply_flow_nominal_m3_h`` (default: sum of the zone maximum flows),
  ``flow_stages_m3_h`` (MULTI_STAGE), ``min_flow_ratio`` (VARIABLE, default 0.3);
* ``operation_schedule``: 24 hourly operating signals f_op;V (default all 1);
* outdoor air: with ``recirculation_control`` "variable" the outdoor air share follows
  (35a), the largest ratio of required outdoor air to zone supply among the zones, so
  every zone gets its ``outdoor_air_flow_m3_h``.  When the flow limits make that
  impossible (zone supply below its requirement, f_ODA;hr < 1, a fixed share or frost
  recirculation) the zone outdoor air falls short: a warning is issued, the hours are
  counted in ``oda_deficit_hours`` and ``q_oda_m3h_<ahu>_<zone>`` reports the outdoor air
  each zone receives;
* ``zones``: list of ``{"zone", "supply_flow_max_m3_h", "supply_flow_min_m3_h",
  "extract_flow_max_m3_h", "outdoor_air_flow_m3_h", "vav"}``.  ``vav`` defaults to True;
  the minimum flow defaults to the outdoor air flow, else 30 % of the maximum; the
  extract defaults to the supply.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

from .ventilation import VentilationStream
from .ventilation_16798_5_1 import (
    SupplyTemperatureControlMode,
    AHUStepInputs,
    AHUStepOutputs,
    ahu_declares_moisture,
    calculate_sensible_ahu_step,
    sensible_ahu_config_from_dict,
)

_RHO_CP = 1.204 * 1006.0          # J/(m3 K), as the zone ventilation streams
AIR_FLOW_CTRL = ("no_ctrl", "on_off", "multi_stage", "variable")


class CentralAHU:
    """A central AHU with its zone terminals (see module docstring)."""

    def __init__(self, cfg: dict, zone_names):
        if not isinstance(cfg, dict):
            raise TypeError("central AHU configuration must be a dict")
        self.name = str(cfg.get("name", "")).strip()
        if not self.name:
            raise ValueError("a central AHU needs a non-empty 'name'")
        zones = cfg.get("zones") or []
        if not zones:
            raise ValueError(f"central AHU {self.name!r}: 'zones' must list the zones it serves")
        known = set(str(z) for z in zone_names)
        self.zones = []
        for zc in zones:
            zn = str(zc.get("zone", ""))
            if zn not in known:
                raise ValueError(f"central AHU {self.name!r}: unknown zone {zn!r}")
            q_max = float(zc["supply_flow_max_m3_h"])
            if not math.isfinite(q_max) or q_max <= 0.0:
                raise ValueError(f"central AHU {self.name!r}, zone {zn!r}: supply_flow_max_m3_h must be > 0")
            q_oda = zc.get("outdoor_air_flow_m3_h")
            q_oda = None if q_oda is None else float(q_oda)
            vav = bool(zc.get("vav", True))
            q_min = zc.get("supply_flow_min_m3_h")
            if q_min is None:
                q_min = q_oda if q_oda is not None else 0.3 * q_max
            q_min = float(min(max(float(q_min), 0.0), q_max))
            q_eta = float(zc.get("extract_flow_max_m3_h", q_max))
            if not math.isfinite(q_eta) or q_eta < 0.0:
                raise ValueError(f"central AHU {self.name!r}, zone {zn!r}: extract_flow_max_m3_h must be >= 0")
            self.zones.append({"zone": zn, "q_max": q_max, "q_min": q_min if vav else q_max,
                               "q_eta_ratio": q_eta / q_max, "q_oda": q_oda, "vav": vav})
        if len({z["zone"] for z in self.zones}) != len(self.zones):
            raise ValueError(f"central AHU {self.name!r}: a zone is listed twice")
        any_vav = any(z["vav"] for z in self.zones)
        self.flow_ctrl = str(cfg.get("air_flow_control", "variable" if any_vav else "on_off")).strip().lower()
        if self.flow_ctrl not in AIR_FLOW_CTRL:
            raise ValueError(f"central AHU {self.name!r}: air_flow_control must be one of {AIR_FLOW_CTRL}")
        q_nom_zones = sum(z["q_max"] for z in self.zones)
        self.q_nom = float(cfg.get("supply_flow_nominal_m3_h", q_nom_zones))
        self.stages = sorted(float(v) for v in (cfg.get("flow_stages_m3_h") or [self.q_nom]))
        if self.flow_ctrl == "multi_stage" and not cfg.get("flow_stages_m3_h"):
            raise ValueError(f"central AHU {self.name!r}: multi_stage needs 'flow_stages_m3_h'")
        self.f_q_min = float(cfg.get("min_flow_ratio", 0.3))
        sched = cfg.get("operation_schedule")
        self.schedule = [1.0] * 24 if sched is None else [float(v) for v in sched]
        if len(self.schedule) != 24 or any(not (0.0 <= v <= 1.0) for v in self.schedule):
            raise ValueError(f"central AHU {self.name!r}: operation_schedule must have 24 values in [0, 1]")
        comp = {k: v for k, v in cfg.items()
                if k not in ("zones", "air_flow_control", "supply_flow_nominal_m3_h",
                             "flow_stages_m3_h", "min_flow_ratio", "operation_schedule")}
        comp.setdefault("supply_flow_m3_h", self.q_nom)
        comp["ventilation_type"] = "mechanical_supply"
        self.config = sensible_ahu_config_from_dict(comp)
        self.declares_moisture = ahu_declares_moisture(comp)
        self.load_compensated = (self.config.supply_temperature_control.mode
                                 is SupplyTemperatureControlMode.LOAD_COMPENSATED)
        if self.load_compensated and any_vav:
            raise ValueError(
                f"central AHU {self.name!r}: a load compensated supply temperature (29c) with VAV "
                "terminals makes both the flow and the temperature load dependent; use constant "
                "volume terminals (vav: false) or another supply temperature control")
        from .ahu_moisture_16798_5_1 import air_density
        self._rho_cp = air_density(self.config.altitude_m) * 1006.0
        # VAV memory: last zone flow and supply temperature at the zone
        self._q_prev = {z["zone"]: z["q_max"] for z in self.zones}
        self.oda_deficit_hours = {z["zone"]: 0 for z in self.zones}
        self._oda_warned = False
        self._t_src_prev = None

    @property
    def zone_names(self):
        return [z["zone"] for z in self.zones]

    # ------------------------------------------------------------------ flows
    def _zone_requirement(self, z, t_air, q_hvac_prev):
        if not z["vav"]:
            return z["q_max"]
        t_src = self._t_src_prev
        if t_src is None:
            return z["q_max"]
        q_prev = self._q_prev[z["zone"]]
        # sensible load of the zone in the previous step (positive: heating)
        load = q_hvac_prev + self._rho_cp * q_prev / 3600.0 * (t_src - t_air)
        dt = t_src - t_air
        if abs(dt) < 0.5 or load * dt <= 0.0:
            return z["q_min"]
        return min(z["q_max"], max(z["q_min"], load / (self._rho_cp * dt) * 3600.0))

    def _ahu_flow(self, q_req, f_op):
        c = self.flow_ctrl
        if c == "no_ctrl":
            return self.q_nom                                              # (23a)
        if c == "on_off":
            return f_op * self.q_nom                                       # (23b)
        if c == "multi_stage":
            for st in self.stages:                                         # (23c)
                if q_req < st:
                    return f_op * min(st, self.q_nom)
            return f_op * min(self.stages[-1], self.q_nom)
        # (23d), limited to the nominal AHU flow (as the EPB Center reference spreadsheet)
        return f_op * min(self.q_nom, max(self.f_q_min * self.q_nom, q_req))

    def _distribute(self, q_ahu, q_req):
        """Zone flows (6) within the terminal maxima.

        Up to the sum of the requirements the AHU flow is shared in proportion to them
        (6).  An AHU flow above the requirements (NO_CTRL, a stage or the minimum flow
        ratio) is shared the same way but no terminal exceeds its maximum: the excess
        goes to the terminals still below it, and the flow above the sum of the maxima
        is not delivered (the terminal dampers limit the network flow).
        """
        q_max = {z["zone"]: z["q_max"] for z in self.zones}
        req_sum = sum(q_req.values())
        if q_ahu <= req_sum or req_sum <= 0.0:
            if req_sum <= 0.0:
                return {zn: 0.0 for zn in q_req}
            return {zn: q_ahu * q_req[zn] / req_sum for zn in q_req}
        q_ahu = min(q_ahu, sum(q_max.values()))
        q = {zn: 0.0 for zn in q_req}
        free = set(q_req)
        left = q_ahu
        while free and left > 1e-9:
            wsum = sum(q_req[zn] for zn in free)
            w = {zn: (q_req[zn] / wsum if wsum > 0.0 else 1.0 / len(free)) for zn in free}
            capped = {zn for zn in free if q[zn] + left * w[zn] >= q_max[zn] - 1e-12}
            if not capped:
                for zn in free:
                    q[zn] += left * w[zn]
                break
            for zn in capped:
                left -= q_max[zn] - q[zn]
                q[zn] = q_max[zn]
            free -= capped
        return q

    # ------------------------------------------------------------------ step
    def step(self, hour_of_day, t_out, zone_air_prev, q_hvac_prev, x_ext=None, zone_x_prev=None,
             hour_of_year=None):
        """One time step.

        ``zone_air_prev``: zone air temperatures of the previous step (extract air and VAV
        sizing); ``q_hvac_prev``: ideal zone system power of the previous step (W, positive
        heating), both of the same step; ``zone_x_prev``: zone moisture contents (moisture processes only when given with
        ``x_ext`` and the AHU declares moisture data).

        Returns ``(outputs, per_zone)`` with the AHU step outputs (whole AHU) and, per
        zone, ``{"q_oda", "q_oda_deficit", "q_req", "q", "q_delivered", "q_extract",
        "q_makeup", "streams", "outputs"}``; ``outputs`` is the AHU output scaled to the zone flow, for the zone
        moisture balance.
        """
        f_op = self.schedule[int(hour_of_day) % 24]
        q_req = {z["zone"]: self._zone_requirement(
                     z, float(zone_air_prev[z["zone"]]), float(q_hvac_prev.get(z["zone"], 0.0)))
                 for z in self.zones}
        q_req_sum = sum(q_req.values())
        q_ahu = self._ahu_flow(q_req_sum, f_op)
        q_zone = self._distribute(q_ahu, q_req)                                # (6)
        q_ahu = sum(q_zone.values())        # the terminals cap the network flow
        share = {zn: (q_zone[zn] / q_ahu if q_ahu > 0.0 else 0.0) for zn in q_zone}
        q_eta = {z["zone"]: q_zone[z["zone"]] * z["q_eta_ratio"] for z in self.zones}  # (7)
        q_eta_sum = sum(q_eta.values())
        if q_eta_sum > 0.0:
            t_eta = sum(q_eta[zn] * float(zone_air_prev[zn]) for zn in q_eta) / q_eta_sum
        else:
            t_eta = sum(float(zone_air_prev[zn]) for zn in q_eta) / len(q_eta)
        humid = self.declares_moisture and x_ext is not None and zone_x_prev is not None
        x_eta = None
        if humid:
            w = q_eta if q_eta_sum > 0.0 else {zn: 1.0 for zn in q_eta}
            x_eta = sum(w[zn] * float(zone_x_prev[zn]) for zn in w) / sum(w.values())
        # (35a): the outdoor air share is set by the most demanding zone,
        # f_ODA;min = max_i(q_ODA;req;i / q_SUP;zv;i); the AHU gets it as a flow
        ratios = [min(1.0, z["q_oda"] / q_zone[z["zone"]]) if q_zone[z["zone"]] > 0.0 else 1.0
                  for z in self.zones if z["q_oda"] is not None and z["q_oda"] > 0.0]
        q_oda_req = max(ratios) * q_ahu if (ratios and q_ahu > 0.0) else None
        # LOAD_COMP (29c): supply temperature that covers the zone loads of the previous
        # step with the current zone flows (flow-weighted over the zones)
        t_sup_req = None
        if self.load_compensated:
            num = den = 0.0
            for z in self.zones:
                zn = z["zone"]
                q_now = q_zone[zn]
                if q_now <= 0.0:
                    continue
                t_air = float(zone_air_prev[zn])
                if self._t_src_prev is None:
                    t_i = t_air
                else:
                    load = (float(q_hvac_prev.get(zn, 0.0))
                            + self._rho_cp * self._q_prev[zn] / 3600.0 * (self._t_src_prev - t_air))
                    t_i = t_air + load / (self._rho_cp * q_now / 3600.0)
                num += q_now * t_i
                den += q_now
            t_sup_req = num / den if den > 0.0 else float(t_eta)
        # fans and flows relative to the nominal AHU flow (74)-(75): the actual flow is the
        # flow fraction of the nominal one
        ff = q_ahu / self.q_nom if self.q_nom > 0.0 else 0.0
        f_v_max = max((q_zone[z["zone"]] / z["q_max"] for z in self.zones), default=0.0)  # (11a)
        out = calculate_sensible_ahu_step(self.config, AHUStepInputs(
            outdoor_temperature_c=float(t_out),
            extract_temperature_c=float(t_eta),
            required_supply_flow_m3_h=float(self.q_nom),
            required_extract_flow_m3_h=float(q_eta_sum / ff) if ff > 0.0 else 0.0,
            flow_fraction=float(min(1.0, ff)),
            outdoor_humidity_ratio=float(x_ext) if humid else None,
            extract_humidity_ratio=float(x_eta) if humid else None,
            required_outdoor_air_flow_m3_h=(q_oda_req / ff if (q_oda_req is not None and ff > 0.0)
                                            else q_oda_req),
            required_supply_temperature_c=t_sup_req,
            hour_of_year=hour_of_year,
            max_zone_flow_ratio=f_v_max,
        ))
        q_del_total = out.actual_supply_flow_m3_h
        t_src = out.actual_supply_temperature_c
        per_zone = {}
        for z in self.zones:
            zn = z["zone"]
            q_del = q_del_total * share[zn] if q_ahu > 0.0 else 0.0
            q_mk = max(0.0, q_eta[zn] - q_del)
            streams = []
            if q_del > 0.0:
                streams.append(VentilationStream(
                    name=f"{self.name}", heat_transfer_coefficient_w_k=self._rho_cp * q_del / 3600.0,
                    source_temperature_c=float(t_src), category="supply"))
            if q_mk > 0.0:
                streams.append(VentilationStream(
                    name=f"{self.name}_makeup", heat_transfer_coefficient_w_k=self._rho_cp * q_mk / 3600.0,
                    source_temperature_c=float(t_out), category="outdoor_air"))
            # outdoor air reaching the zone: its share of the supply plus the make-up air
            q_oda_zone = out.outdoor_air_fraction * q_del + q_mk
            deficit = (max(0.0, z["q_oda"] - q_oda_zone)
                       if (z["q_oda"] is not None and q_ahu > 0.0) else 0.0)
            if deficit > 1e-6 * max(1.0, z["q_oda"] or 0.0):
                self.oda_deficit_hours[zn] += 1
                if not self._oda_warned:
                    self._oda_warned = True
                    warnings.warn(
                        f"central AHU {self.name!r}: zone {zn!r} receives "
                        f"{q_oda_zone:.0f} m3/h of outdoor air, less than the required "
                        f"{z['q_oda']:.0f} m3/h (zone supply {q_del:.0f} m3/h, outdoor air "
                        f"share {out.outdoor_air_fraction:.2f}); the flow limits make the "
                        "requirement impossible. See oda_deficit_hours and "
                        f"q_oda_m3h_{self.name}_<zone>.", stacklevel=3)
            per_zone[zn] = {
                "q_oda": q_oda_zone, "q_oda_deficit": deficit,
                "q_req": q_req[zn], "q": q_zone[zn], "q_delivered": q_del,
                "q_extract": q_eta[zn], "q_makeup": q_mk, "streams": streams,
                "outputs": dataclasses.replace(out, actual_supply_flow_m3_h=q_del,
                                               actual_extract_flow_m3_h=q_eta[zn],
                                               makeup_outdoor_air_flow_m3_h=q_mk),
            }
        # VAV memory for the next step
        for zn in q_zone:
            self._q_prev[zn] = q_zone[zn] if q_ahu > 0.0 else 0.0
        self._t_src_prev = float(t_src) if q_ahu > 0.0 else self._t_src_prev
        return out, per_zone


def central_ahus_from_building(building_object, zone_names):
    """Central AHUs declared in ``building_parameters.central_ahus`` (empty list if none)."""
    if not isinstance(building_object, dict):
        return []
    raw = (building_object.get("building_parameters") or {}).get("central_ahus") or []
    ahus = [CentralAHU(c, zone_names) for c in raw]
    names = [a.name for a in ahus]
    if len(set(names)) != len(names):
        raise ValueError("central AHU names must be unique")
    served = [zn for a in ahus for zn in a.zone_names]
    if len(set(served)) != len(served):
        raise ValueError("a zone can be served by one central AHU only")
    return ahus


__all__ = ["CentralAHU", "central_ahus_from_building", "AIR_FLOW_CTRL", "AHUStepOutputs"]
