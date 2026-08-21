from copy import deepcopy
import numpy as np
from pybuildingenergy.global_inputs import TB14 as TB14_backup
import pandas as pd
from pybuildingenergy.source.emitter_442 import (
    EN442RadiatorCharacteristic,
    EmitterCharacteristic,
    normalize_emitter_characteristic_method,
)
from pybuildingenergy.source.table_iso_16798_1 import internal_gains_occupants

def _dir_from_orientation(azimuth: float, tilt: float) -> str:
    '''
    Map an orientation to a cardianl code used for adjacency logic

    Args:
        azimuth (float): azimuth angle of the surface
        tilt (float): tilt angle of the surface

    Returns:
        str: one of "HOR"(horizontal), "NV"(north vertical), "EV"(east vertical), "SV"(south vertical), "WV"(west vertical)
    '''
    # 0=N, 90=E, 180=S, 270=W; tilt 90 = verticale; 0 = orizzontale
    if abs(float(tilt) - 0.0) <= 1e-9:
        return "HOR"
    if abs(float(tilt) - 90.0) > 1e-6:
        # fallback: soglia 45°
        return "HOR" if float(tilt) < 45.0 else "NV"
    az = float(azimuth) % 360.0
    if abs(az-0.0) <= 1e-6 or abs(az-360.0) <= 1e-6: return "NV"
    if abs(az-90.0) <= 1e-6:  return "EV"
    if abs(az-180.0) <= 1e-6: return "SV"
    if abs(az-270.0) <= 1e-6: return "WV"
    # snap to nearest
    pts  = [0.0, 90.0, 180.0, 270.0]
    labs = ["NV","EV","SV","WV"]
    return labs[int(np.argmin(np.abs((az - np.array(pts)) % 360.0)))]


def _adj_op_area_in_dir(adj_zone: dict, dir_code: str) -> float:
    A   = np.asarray(adj_zone["area_facade_elements"], dtype=float)
    Typ = np.asarray(adj_zone["typology_elements"], dtype=object)
    Ori = np.asarray(adj_zone["orientation_elements"], dtype=object)
    mask = (np.char.upper(Typ.astype(str)) == "OP") & (np.char.upper(Ori.astype(str)) == dir_code)
    return float(A[mask].sum()) if mask.any() else 0.0

#  -------------------------------------------------------------------
#  CHECK INPUT HEATING SYSTEM
# -------------------------------------------------------------------



def check_heating_system_inputs(system_input: dict):
    """
    Upload and validate the heating system configuration from a dict.

    Rules:
    - If 'TB14' is present and DataFrame -> use that.
    - Otherwise -> use the backup TB14 from global_inputs.py.
    - If emitter_type is not an index of the TB14 selected:
        -> auto-set to the first available index and emit a warning message.
    - If gen_flow_temp_control_type = Type A, then gen_outdoor_temp_data must be provided as a dataframe.
    - If gen_flow_temp_control_type = Type C, if θHW_gen_flw_const is not provided, it is set to 50.

    Returns:
        {
            "TB14_used": DataFrame,
            "emitter_type": str,
            "messages": list[str],
            "config": dict  # input normalized with updated emitter_type
        }
    """
    messages = []
    cfg = deepcopy(system_input)

    # 1) TB14: custom or default
    tb14_custom = cfg.get("TB14", None)
    if isinstance(tb14_custom, pd.DataFrame):
        TB14 = tb14_custom
        messages.append("✅ Custom TB14 table loaded from input.")
    else:
        TB14 = TB14_backup
        messages.append("⚙️ Default TB14 table loaded from global_inputs.py.")

    required_tb14_columns = {
        "Emitters_nominale_deltaTeta_air_C",
        "Emitters_exponent_n",
        "Emitters_nominal_deltaTeta_Water_C",
    }
    missing_tb14_columns = required_tb14_columns.difference(TB14.columns)
    if missing_tb14_columns:
        raise ValueError(
            "TB14 is missing required columns: "
            + ", ".join(sorted(missing_tb14_columns))
        )
    numeric_tb14 = TB14.loc[:, sorted(required_tb14_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_tb14.isna().any().any() or (numeric_tb14 <= 0.0).any().any():
        raise ValueError("All TB14 emitter parameters must be finite positive numbers.")

    # 2) Validate emitter_type against TB14 or an explicit EN 442 rating.
    emitter_type = cfg.get("emitter_type", None)
    valid_emitters = list(TB14.index)

    supplied_characteristic = cfg.get("emitter_characteristic")
    configured_method = cfg.get(
        "emitter_calculation_method",
        cfg.get("emitter_characteristic_method"),
    )
    if configured_method is None and isinstance(
        supplied_characteristic, EmitterCharacteristic
    ):
        configured_method = supplied_characteristic.method
    method = normalize_emitter_characteristic_method(configured_method)
    cfg["emitter_calculation_method"] = method

    if isinstance(supplied_characteristic, EmitterCharacteristic):
        supplied_method = normalize_emitter_characteristic_method(
            supplied_characteristic.method
        )
        if supplied_method != method:
            raise ValueError(
                "emitter_characteristic.method conflicts with "
                "emitter_calculation_method."
            )

    if method == "en442":
        if emitter_type is None:
            emitter_type = "Radiator"
            cfg["emitter_type"] = emitter_type
        emitter_label = str(emitter_type).strip().lower()
        excluded = ("floor", "fan coil", "fan-coil", "fan assisted", "trench")
        if any(label in emitter_label for label in excluded):
            raise ValueError(
                "EN 442 is not applicable to floor heating, fan-assisted emitters, "
                "or trench convectors."
            )
        rating = cfg.get("emitter_rating")
        if isinstance(supplied_characteristic, EmitterCharacteristic):
            validated_rating = supplied_characteristic
        elif isinstance(rating, dict):
            validated_rating = EN442RadiatorCharacteristic.from_dict(rating)
        else:
            raise ValueError(
                "emitter_calculation_method='en442' requires emitter_rating "
                "or an EN 442 emitter_characteristic object."
            )
        messages.append(
            "EN 442 product rating validated: "
            f"phi_50={validated_rating.nominal_power_kW:g} kW, "
            f"n={validated_rating.exponent_n:g}."
        )
        if validated_rating.maximum_operating_temperature_C is None:
            messages.append(
                "EN 442 product rating has no maximum operating temperature; "
                "temperature-based capacity limiting will be unavailable."
            )
    elif method == "custom":
        characteristic = cfg.get("emitter_characteristic", cfg.get("emitter_rating"))
        if isinstance(characteristic, EmitterCharacteristic):
            missing = []
        elif not isinstance(characteristic, dict):
            raise ValueError(
                "emitter_calculation_method='custom' requires an "
                "emitter_characteristic object or dictionary."
            )
        else:
            missing = []
            if (
                characteristic.get("nominal_excess_temperature_K") is None
                and characteristic.get("rated_excess_temperature_K") is None
            ):
                missing.append("nominal_excess_temperature_K")
            if (
                characteristic.get("exponent_n") is None
                and characteristic.get("rated_exponent_n") is None
            ):
                missing.append("exponent_n")
        if missing:
            raise ValueError(
                "Custom emitter characteristic is missing: " + ", ".join(missing)
            )
        if emitter_type is None:
            cfg["emitter_type"] = "Custom emitter"
        messages.append("Custom emitter characteristic validated.")
    elif emitter_type not in valid_emitters:
        fallback = valid_emitters[0] if valid_emitters else None
        messages.append(
            f"⚠️ Emitter type '{emitter_type}' not found in TB14; auto-set to '{fallback}'."
        )
        cfg["emitter_type"] = fallback
    else:
        messages.append(f"✅ Emitter type '{emitter_type}' found in TB14.")

    # 3) Validazione gen_flow_temp_control_type (Type A, B, or C)
    # NB: allow descriptive suffixes such as 'Type A - Based on outdoor temperature'
    # The downstream code in iso_15316_1.calculate_circuit_node_temperature uses startswith().
    gen_flow_temp_control_type = cfg.get("gen_flow_temp_control_type", None)
    _ctrl_str = str(gen_flow_temp_control_type) if gen_flow_temp_control_type is not None else ''

    if _ctrl_str.startswith('Type A'):
        gen_outdoor_temp_data = cfg.get('gen_outdoor_temp_data', None)
        if not isinstance(gen_outdoor_temp_data, pd.DataFrame):
            messages.append("⚠️ For 'Type A' control, 'gen_outdoor_temp_data' must be provided as a DataFrame.")
            cfg["gen_flow_temp_control_type"] = 'Type B'  # Auto-switch to Type B
        else:
            messages.append("✅ 'gen_outdoor_temp_data' provided for 'Type A' control.")

    elif _ctrl_str.startswith('Type C'):
        if 'θHW_gen_flw_const' not in cfg or cfg['θHW_gen_flw_const'] is None:
            messages.append("⚠️ 'θHW_gen_flw_const' not provided for 'Type C' control; setting it to 50.")
            cfg['θHW_gen_flw_const'] = 50.0
        else:
            messages.append(f"✅ 'θHW_gen_flw_const' provided: {cfg['θHW_gen_flw_const']} for 'Type C' control.")

    elif not _ctrl_str.startswith('Type B'):
        messages.append(f"⚠️ Invalid value for 'gen_flow_temp_control_type': '{gen_flow_temp_control_type}'. Setting to 'Type B'.")
        cfg["gen_flow_temp_control_type"] = 'Type B'

    return {
        "TB14_used": TB14,
        "emitter_type": cfg["emitter_type"],
        "messages": messages,
        "config": cfg
    }




def sanitize_and_validate_BUI(bui: dict,
                              fix: bool = True,
                              eps: float = 1e-6,
                              defaults: dict | None = None):
    """
    Validates and (optionally) sanitizes a BUI, with extra checks:
    - Check area OP (NV/EV/SV/WV) of the adjacent zone with respect to the BUI wall 'adjacent';
    - Check windows in the same direction of the 'adjacent' wall (windows cannot exceed
      the difference between the area of the BUI 'adjacent' and the area of the OP of the adjacent zone);
    - If the adjacent zone covers less area than the BUI 'adjacent', creates a new
      'opaque' element with area equal to the difference and resizes the 'adjacent' wall to the sole area covered.
    """

    bui_clean = deepcopy(bui)
    issues = []
    D = defaults or {
        "opaque_u_default": 0.5,
        "transparent_u_default": 1.6,
        "g_default": 0.6
    }

    def add_issue(level, path, msg, fixed=False):
        issues.append({"level": level, "path": path, "msg": msg, "fix_applied": fixed})

    # ---------------------------
    # 1) BUILDING-LEVEL CHECKS
    # ---------------------------
    b = bui_clean.get("building", {})
    for key in ["net_floor_area", "exposed_perimeter", "height"]:
        if key in b:
            val = b[key]
            if not (isinstance(val, (int, float)) and val > 0):
                if fix and isinstance(val, (int, float)) and val == 0:
                    b[key] = max(eps, 1.0)
                    add_issue("WARN", f"building.{key}", f"{key} was 0; set to {b[key]}", fixed=True)
                else:
                    add_issue("ERROR", f"building.{key}", f"{key} should be > 0; value={val}", fixed=False)

    if "n_floors" in b and (not isinstance(b["n_floors"], (int, float)) or b["n_floors"] <= 0):
        if fix and isinstance(b["n_floors"], (int, float)) and b["n_floors"] == 0:
            b["n_floors"] = 1
            add_issue("WARN", "building.n_floors", "n_floors was 0; set to 1", fixed=True)
        else:
            add_issue("ERROR", "building.n_floors", f"n_floors should be > 0; value={b.get('n_floors')}", fixed=False)

    # building_type_class validation against internal_gains_occupants table
    btc = b.get("building_type_class", None)
    if btc is not None:
        allowed_classes = set(internal_gains_occupants.keys())
        fallback_class = "Residential_apartment"
        if btc not in allowed_classes:
            msg = (
                f"building_type_class '{btc}' not recognized; set to '{fallback_class}' "
                f"and continuing with the fallback."
            )
            b["building_type_class"] = fallback_class
            print(f"⚠️ {msg}")
            add_issue("WARN", "building.building_type_class", msg, fixed=True)

    # ---------------------------
    # 2) SURFACES CHECKS
    # ---------------------------
    type_corrections = {"opque": "opaque", "opaqu": "opaque", "trasparent": "transparent"}
    allowed_types = {"opaque", "transparent", "adiabatic", "adjacent"}

    for i, s in enumerate(bui_clean.get("building_surface", [])):
        path = f"building_surface[{i}]"
        # typo correction for "type"
        t = s.get("type")
        if isinstance(t, str) and t.lower() in type_corrections:
            old = t
            s["type"] = type_corrections[t.lower()]
            add_issue("WARN", f"{path}.type", f'type="{old}" corrected to "{s["type"]}"', fixed=True)
        t = (s.get("type") or "").lower()
        if t not in allowed_types:
            add_issue("ERROR", f"{path}.type", f'type "{s.get("type")}" not recognized', fixed=False)

        # area
        area = s.get("area", None)
        if not (isinstance(area, (int, float)) and area > 0):
            if fix and isinstance(area, (int, float)) and area == 0:
                s["area"] = max(eps, 1.0)
                add_issue("WARN", f"{path}.area", f"area was 0; set to {s['area']}", fixed=True)
            else:
                add_issue("ERROR", f"{path}.area", f"area should be > 0; value={area}", fixed=False)

        # sky_view_factor
        svf = s.get("sky_view_factor", None)
        if not (isinstance(svf, (int, float)) and 0.0 <= svf <= 1.0):
            if fix and isinstance(svf, (int, float)):
                s["sky_view_factor"] = min(1.0, max(0.0, svf))
                add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor out of range; clipped to {s['sky_view_factor']}", fixed=True)
            else:
                add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor missing or not in [0,1]; values={svf}", fixed=False)

        # U-value for opaque/transparent
        if t in {"opaque", "transparent"}:
            u = s.get("u_value", None)
            if not (isinstance(u, (int, float)) and u > 0):
                if fix:
                    default_u = D["opaque_u_default"] if t == "opaque" else D["transparent_u_default"]
                    s["u_value"] = default_u
                    add_issue("WARN", f"{path}.u_value", f"u_value invalid ({u}); set to default value: {default_u}", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.u_value", f"u_value should be > 0; value={u}", fixed=False)

        # g-value for transparent
        if t == "transparent":
            g = s.get("g_value", None)
            if not (isinstance(g, (int, float)) and g > 0):
                if fix:
                    s["g_value"] = D["g_default"]
                    add_issue("WARN", f"{path}.g_value", f"g_value invalid ({g}); set to default value {D['g_default']}", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.g_value", f"g_value should be > 0; value={g}", fixed=False)

            # window dimensions
            h = s.get("height", None)
            w = s.get("width", None)
            if not (isinstance(h, (int, float)) and h > 0):
                if fix:
                    s["height"] = 1.0
                    add_issue("WARN", f"{path}.height", f"height not valid ({h}); set to 1.0", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.height", f"height should be > 0; value={h}", fixed=False)
            if not (isinstance(w, (int, float)) and w > 0):
                if fix:
                    s["width"] = 1.0
                    add_issue("WARN", f"{path}.width", f"width not valid ({w}); set to 1.0", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.width", f"width should be > 0; value={w}", fixed=False)

            # parapet 0..height
            para = s.get("parapet", None)
            if para is not None and isinstance(para, (int, float)) and h is not None and isinstance(h, (int,float)):
                if not (0.0 <= para <= max(h, 0.0)):
                    if fix:
                        s["parapet"] = min(max(0.0, para if isinstance(para, (int,float)) else 0.0), h if isinstance(h,(int,float)) else 1.0)
                        add_issue("WARN", f"{path}.parapet", f"parapet out of range; clipped to {s['parapet']}", fixed=True)
                    else:
                        add_issue("WARN", f"{path}.parapet", f"parapet out of range (0..height); value={para}, height={h}", fixed=False)

        # recommended orientations
        ori = s.get("orientation", {})
        tilt = ori.get("tilt", None)
        az   = ori.get("azimuth", None)
        if not (isinstance(tilt, (int, float)) and tilt in (0, 90)):
            if fix and isinstance(tilt, (int, float)):
                s["orientation"]["tilt"] = 0 if abs(tilt - 0) < abs(tilt - 90) else 90
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt non-standard ({tilt}); normalized to {s['orientation']['tilt']}", fixed=True)
            else:
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt recommended 0 or 90; value={tilt}", fixed=False)
        if not (isinstance(az, (int, float)) and az in (0, 90, 180, 270)):
            if fix and isinstance(az, (int, float)):
                snapped = int(round(az / 90.0) * 90) % 360
                if snapped == 360: snapped = 0
                s["orientation"]["azimuth"] = snapped
                add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth non-standard ({az}); normalized to {snapped}", fixed=True)
            else:
                add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth recommended in {{0,90,180,270}}; value={az}", fixed=False)

    # ---------------------------
    # 3) ADJACENT ZONES CHECKS
    # ---------------------------
    for j, az in enumerate(bui_clean.get("adjacent_zones", [])):
        path = f"adjacent_zones[{j}]"
        vol = az.get("volume", None)
        if not (isinstance(vol, (int, float)) and vol > 0):
            if fix and isinstance(vol, (int, float)) and vol == 0:
                az["volume"] = 1.0
                add_issue("WARN", f"{path}.volume", "volume era 0; set to 1.0", fixed=True)
            else:
                add_issue("ERROR", f"{path}.volume", f"volume should be > 0; value={vol}", fixed=False)

        # length coherence
        arrs = {
            "area_facade_elements": az.get("area_facade_elements"),
            "typology_elements": az.get("typology_elements"),
            "transmittance_U_elements": az.get("transmittance_U_elements"),
            "orientation_elements": az.get("orientation_elements"),
        }
        lengths = [len(v) for v in arrs.values() if isinstance(v, (list, np.ndarray))]
        if lengths and len(set(lengths)) != 1:
            add_issue("ERROR", path, f"Array lengths are inconsistent: {[ (k, None if v is None else len(v)) for k,v in arrs.items() ]}", fixed=False)

        # U elementi > 0
        U = arrs.get("transmittance_U_elements")
        if isinstance(U, (list, np.ndarray)):
            U = np.asarray(U, dtype=float)
            bad = np.where(U <= 0)[0].tolist()
            if bad:
                if fix:
                    U[U <= 0] = D["opaque_u_default"]
                    az["transmittance_U_elements"] = U
                    add_issue("WARN", f"{path}.transmittance_U_elements", f"U<=0 at idx {bad}; set to {D['opaque_u_default']}", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.transmittance_U_elements", f"U<=0 ai idx {bad}", fixed=False)
    
    # -------------------------------------------------------------------
    # 4) ADJACENCY PAIRING CAPS + FINESTRE + SPLIT ESTERNO
    # -------------------------------------------------------------------
    def _azimuth_to_dir(az: float) -> str | None:
        if az is None:
            return None
        azn = int(round(float(az))) % 360
        if azn in (0, 360): return "NV"
        if azn == 90:       return "EV"
        if azn == 180:      return "SV"
        if azn == 270:      return "WV"
        return None

    adj_by_name = { z.get("name"): z for z in bui_clean.get("adjacent_zones", []) }

    i = 0
    while i < len(bui_clean.get("building_surface", [])):
        s = bui_clean["building_surface"][i]
        path = f"building_surface[{i}]"

        if (s.get("type") != "adjacent") or (not s.get("name_adj_zone")):
            i += 1
            continue

        ori = s.get("orientation", {}) or {}
        tilt = ori.get("tilt", None)
        az   = ori.get("azimuth", None)

        # Only consider vertical walls
        if not (isinstance(tilt, (int,float)) and abs(tilt-90) < 1e-6):
            i += 1
            continue

        dir_card = _azimuth_to_dir(az)
        if dir_card is None:
            i += 1
            continue

        A_bui_orig = float(s.get("area", 0.0))
        if A_bui_orig <= 0:
            i += 1
            continue

        adj_name = s["name_adj_zone"]
        if adj_name not in adj_by_name:
            add_issue("ERROR", path, f"name_adj_zone='{adj_name}' not found in adjacent_zones", fixed=False)
            i += 1
            continue

        # --- adjacent zone arrays (writable) ---
        azone = adj_by_name[adj_name]
        A_arr  = np.array(azone.get("area_facade_elements", []), dtype=float).copy()
        TypArr = np.array(azone.get("typology_elements", []),     dtype=object)
        OriArr = np.array(azone.get("orientation_elements", []),  dtype=object)

        if not (len(A_arr) == len(TypArr) == len(OriArr)):
            add_issue("ERROR", f"adjacent_zones[{adj_name}]", "Arrays are inconsistent (len mismatch) for direction cap", fixed=False)
            i += 1
            continue

        # === (1) Cap OP area of the adjacent zone in the direction ===
        mask_op_dir = (np.char.upper(TypArr.astype(str)) == "OP") & (np.char.upper(OriArr.astype(str)) == dir_card)
        idx = np.where(mask_op_dir)[0]
        if idx.size == 0:
            A_adj_dir_cap = 0.0
        else:
            A_adj_dir_cap = float(A_arr[idx].sum())
            if A_adj_dir_cap > A_bui_orig + eps:
                msg = (f"Zone '{adj_name}' OP {dir_card}={A_adj_dir_cap:.3f} m² > BUI wall {A_bui_orig:.3f} m² "
                       f"({s.get('name','(no name)')})")
                if not fix:
                    add_issue("ERROR", path, msg + " (no fix)", fixed=False)
                else:
                    sf = A_bui_orig / max(A_adj_dir_cap, eps)
                    A_arr[idx] = A_arr[idx] * sf
                    azone["area_facade_elements"] = A_arr  # write-back
                    A_adj_dir_cap = float(A_arr[idx].sum())
                    add_issue("WARN", path, msg + f" → reduced (scale {sf:.3f})", fixed=True)

        # === (2) WINDOW CHECK in the same direction (uses A_bui_orig and A_adj_dir_cap) ===
        win_idx = []
        for k, ws in enumerate(bui_clean.get("building_surface", [])):
            if (ws.get("type") == "transparent"
                and isinstance(ws.get("orientation",{}).get("tilt", None), (int,float))
                and abs(ws["orientation"]["tilt"] - 90) < 1e-6):
                if _azimuth_to_dir(ws["orientation"].get("azimuth", None)) == dir_card:
                    win_idx.append(k)

        if win_idx:
            A_win_dir = float(sum(bui_clean["building_surface"][k].get("area", 0.0) for k in win_idx))
            diff = max(A_bui_orig - A_adj_dir_cap, 0.0)

            if A_win_dir > diff + eps:
                msg = (f"Windows {dir_card}: window area={A_win_dir:.3f} m² > difference "
                       f"(A_BUI_adjacent - A_adj_OP)={diff:.3f} m² on {s.get('name','(no name)')}")
                if not fix:
                    add_issue("ERROR", path, msg + " (no fix)", fixed=False)
                else:
                    # scale all windows in this direction proportionally so they sum to 'diff'
                    sfw = (diff / max(A_win_dir, eps)) if diff > 0 else 0.0
                    for k in win_idx:
                        oldA = float(bui_clean["building_surface"][k].get("area", 0.0))
                        newA = oldA * sfw
                        bui_clean["building_surface"][k]["area"] = newA
                    add_issue("WARN", path, msg + f" → window areas reduced (scale {sfw:.3f})", fixed=True)

        # === (3) SPLIT: if the adjacent zone covers less than A_bui_orig, create new 'opaque' wall for the residual ===
        diff_ext = max(A_bui_orig - A_adj_dir_cap, 0.0)
        if diff_ext > eps:
            if not fix:
                add_issue("ERROR", path, (f"'adjacent' wall {s.get('name','(no name)')} not fully covered by "
                                          f"zone '{adj_name}' ({A_adj_dir_cap:.3f} < {A_bui_orig:.3f}). "
                                          f"Residual {diff_ext:.3f} m² not allocated."), fixed=False)
            else:
                # create new OPAQUE element for the residual
                new_surf = {
                    "name": f"{s.get('name','(adjacent)')} — external residual",
                    "type": "opaque",
                    "area": float(diff_ext),
                    "sky_view_factor": s.get("sky_view_factor", 0.5),
                    "u_value": s.get("u_value", D["opaque_u_default"]),
                    "solar_absorptance": s.get("solar_absorptance", 0.6),
                    "thermal_capacity": s.get("thermal_capacity", 1000),
                    "orientation": {
                        "azimuth": s.get("orientation", {}).get("azimuth", 0),
                        "tilt": s.get("orientation", {}).get("tilt", 90),
                    },
                    "name_adj_zone": None,
                }
                bui_clean["building_surface"].append(new_surf)

                # resize the adjacent wall to only the area covered by the adjacent zone
                s["area"] = float(A_adj_dir_cap)

                add_issue("WARN", path, (f"Created 'opaque' residual of {diff_ext:.3f} m² and "
                                         f"resized 'adjacent' wall to {A_adj_dir_cap:.3f} m²."), fixed=True)

        # move to next element (list may have grown)
        i += 1

        # ----------------------------------------------------------
        # 4) SPLIT ADJACENT WALLS: adjacent part + external part
        #    Create a new "opaque" surface for the external portion.
        #    (EXECUTE AFTER ALL CHECKS)
        # ----------------------------------------------------------
        new_surfaces = []   # collect here and append with .extend at end of loop
        for i, s in enumerate(bui_clean.get("building_surface", [])):
            if (s.get("type") != "adjacent") or (not s.get("name_adj_zone")):
                continue

            # Direction NV/EV/SV/WV/HOR of the BUI surface
            az = s.get("orientation", {}).get("azimuth", 0.0)
            tl = s.get("orientation", {}).get("tilt", 90.0)
            dir_code = _dir_from_orientation(az, tl)

            # Only consider vertical walls for splitting with adjacent zones
            if dir_code not in ("NV","EV","SV","WV"):
                continue

            A_bui = float(s.get("area", 0.0))
            if A_bui <= 0:
                continue

            # Find the matching adjacent zone by name
            adj_name = s.get("name_adj_zone")
            adj_zone = None
            for azj in bui_clean.get("adjacent_zones", []):
                if azj.get("name") == adj_name:
                    adj_zone = azj
                    break
            if adj_zone is None:
                # Name not found: log it
                add_issue("WARN", f"building_surface[{i}]", f'name_adj_zone="{adj_name}" not found among adjacent_zones', fixed=False)
                continue

            # OP area of the adjacent zone in the same direction
            A_adj_dir = _adj_op_area_in_dir(adj_zone, dir_code)

            # CAP: adjacent zone cannot exceed the BUI wall
            if A_adj_dir > A_bui:
                add_issue("WARN",
                        f"building_surface[{i}]",
                        f"Adjacent zone area {adj_name} in {dir_code} ({A_adj_dir:.3f} m²) > BUI area ({A_bui:.3f} m²); capped to {A_bui:.3f}",
                        fixed=True)
                A_adj_dir = A_bui

            # If the adjacent zone is smaller than the BUI wall → create external residual
            diff = A_bui - A_adj_dir
            if diff > 1e-9:
                # 1) shrink the ADJACENT surface to only the part in contact with the unconditioned zone
                old_area = s["area"]
                s["area"] = A_adj_dir

                # 2) create a NEW external "opaque" surface with area = diff
                new_surf = {
                    "name": f'{s.get("name","Adjacent split")} — external residual',
                    "type": "opaque",
                    "area": diff,
                    "sky_view_factor": s.get("sky_view_factor", 0.5),
                    "u_value": s.get("u_value", 1.4),
                    "solar_absorptance": s.get("solar_absorptance", 0.6),
                    "thermal_capacity": s.get("thermal_capacity", 0.0),
                    "orientation": {
                        "azimuth": az,
                        "tilt": tl,
                    },
                    "name_adj_zone": None,   # critical: exterior, not adjacent
                }

                # if conv./rad. coefficients are already hydrated, copy them (optional)
                for k in (
                    "convective_heat_transfer_coefficient_internal",
                    "radiative_heat_transfer_coefficient_internal",
                    "convective_heat_transfer_coefficient_external",
                    "radiative_heat_transfer_coefficient_external",
                ):
                    if k in s: new_surf[k] = s[k]

                new_surfaces.append(new_surf)
                add_issue("INFO",
                        f"building_surface[{i}]",
                        f"Split {dir_code}: reduced ADJ area from {old_area:.3f} to {A_adj_dir:.3f} m²; created new external OPAQUE of {diff:.3f} m².",
                        fixed=True)

        # Append all new surfaces
        if new_surfaces:
            bui_clean["building_surface"].extend(new_surfaces)


    return bui_clean, issues


    
