from copy import deepcopy
import numpy as np
from pybuildingenergy.global_inputs import TB14 as TB14_backup

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
    # snap al più vicino
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

import pandas as pd

def check_heating_system_inputs(system_input: dict):
    """
    Upload and validate the heating system configuration from a dict.

    Rules:
    - If 'TB14' is present and DataFrame -> use that.
    - Otherwise -> use the backup TB14 from global_inputs.py.
    - If emitter_type is not an index of the TB14 selected:
        -> set automatico al primo indice disponibile e messaggio di avviso.
    - If gen_flow_temp_control_type = Type A, then gen_outdoor_temp_data must be provided as a dataframe.
    - If gen_flow_temp_control_type = Type C, if θHW_gen_flw_const is not provided, it is set to 50.

    Returns:
        {
            "TB14_used": DataFrame,
            "emitter_type": str,
            "messages": list[str],
            "config": dict  # input normalizzato con emitter_type aggiornato
        }
    """
    messages = []
    cfg = deepcopy(system_input)

    # 1) TB14: custom o default
    tb14_custom = cfg.get("TB14", None)
    if isinstance(tb14_custom, pd.DataFrame):
        TB14 = tb14_custom
        messages.append("✅ Custom TB14 table loaded from input.")
    else:
        TB14 = TB14_backup
        messages.append("⚙️ Default TB14 table loaded from global_inputs.py.")

    # 2) Validazione emitter_type rispetto alla TB14 usata
    emitter_type = cfg.get("emitter_type", None)
    valid_emitters = list(TB14.index)

    if emitter_type not in valid_emitters:
        fallback = valid_emitters[0] if valid_emitters else None
        messages.append(
            f"⚠️ Emitter type '{emitter_type}' not found in TB14; auto-set to '{fallback}'."
        )
        cfg["emitter_type"] = fallback
    else:
        messages.append(f"✅ Emitter type '{emitter_type}' found in TB14.")

    # 3) Validazione gen_flow_temp_control_type (Type A, B, or C)
    gen_flow_temp_control_type = cfg.get("gen_flow_temp_control_type", None)
    
    if gen_flow_temp_control_type == 'Type A':
        gen_outdoor_temp_data = cfg.get('gen_outdoor_temp_data', None)
        if not isinstance(gen_outdoor_temp_data, pd.DataFrame):
            messages.append("⚠️ For 'Type A' control, 'gen_outdoor_temp_data' must be provided as a DataFrame.")
            cfg["gen_flow_temp_control_type"] = 'Type B'  # Auto-switch to Type B
        else:
            messages.append("✅ 'gen_outdoor_temp_data' provided for 'Type A' control.")
    
    elif gen_flow_temp_control_type == 'Type C':
        if 'θHW_gen_flw_const' not in cfg or cfg['θHW_gen_flw_const'] is None:
            messages.append("⚠️ 'θHW_gen_flw_const' not provided for 'Type C' control; setting it to 50.")
            cfg['θHW_gen_flw_const'] = 50.0
        else:
            messages.append(f"✅ 'θHW_gen_flw_const' provided: {cfg['θHW_gen_flw_const']} for 'Type C' control.")
    
    elif gen_flow_temp_control_type != 'Type B':
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
                    add_issue("WARN", f"building.{key}", f"{key} era 0; impostato a {b[key]}", fixed=True)
                else:
                    add_issue("ERROR", f"building.{key}", f"{key} deve essere > 0; valore={val}", fixed=False)

    if "n_floors" in b and (not isinstance(b["n_floors"], (int, float)) or b["n_floors"] <= 0):
        if fix and isinstance(b["n_floors"], (int, float)) and b["n_floors"] == 0:
            b["n_floors"] = 1
            add_issue("WARN", "building.n_floors", "n_floors era 0; impostato a 1", fixed=True)
        else:
            add_issue("ERROR", "building.n_floors", f"n_floors deve essere > 0; valore={b.get('n_floors')}", fixed=False)

    # ---------------------------
    # 2) SURFACES CHECKS
    # ---------------------------
    type_corrections = {"opque": "opaque", "opaqu": "opaque", "trasparent": "transparent"}
    allowed_types = {"opaque", "transparent", "adiabatic", "adjacent"}

    for i, s in enumerate(bui_clean.get("building_surface", [])):
        path = f"building_surface[{i}]"
        # correzione refusi type
        t = s.get("type")
        if isinstance(t, str) and t.lower() in type_corrections:
            old = t
            s["type"] = type_corrections[t.lower()]
            add_issue("WARN", f"{path}.type", f'type="{old}" corretto in "{s["type"]}"', fixed=True)
        t = (s.get("type") or "").lower()
        if t not in allowed_types:
            add_issue("ERROR", f"{path}.type", f'type "{s.get("type")}" non riconosciuto', fixed=False)

        # area
        area = s.get("area", None)
        if not (isinstance(area, (int, float)) and area > 0):
            if fix and isinstance(area, (int, float)) and area == 0:
                s["area"] = max(eps, 1.0)
                add_issue("WARN", f"{path}.area", f"area era 0; impostata a {s['area']}", fixed=True)
            else:
                add_issue("ERROR", f"{path}.area", f"area deve essere > 0; valore={area}", fixed=False)

        # sky_view_factor
        svf = s.get("sky_view_factor", None)
        if not (isinstance(svf, (int, float)) and 0.0 <= svf <= 1.0):
            if fix and isinstance(svf, (int, float)):
                s["sky_view_factor"] = min(1.0, max(0.0, svf))
                add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor fuori range; clippato a {s['sky_view_factor']}", fixed=True)
            else:
                add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor mancante o non in [0,1]; valore={svf}", fixed=False)

        # U-value per opaque/transparent
        if t in {"opaque", "transparent"}:
            u = s.get("u_value", None)
            if not (isinstance(u, (int, float)) and u > 0):
                if fix:
                    default_u = D["opaque_u_default"] if t == "opaque" else D["transparent_u_default"]
                    s["u_value"] = default_u
                    add_issue("WARN", f"{path}.u_value", f"u_value non valido ({u}); impostato default {default_u}", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.u_value", f"u_value deve essere > 0; valore={u}", fixed=False)

        # g-value per trasparenti
        if t == "transparent":
            g = s.get("g_value", None)
            if not (isinstance(g, (int, float)) and g > 0):
                if fix:
                    s["g_value"] = D["g_default"]
                    add_issue("WARN", f"{path}.g_value", f"g_value non valido ({g}); impostato default {D['g_default']}", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.g_value", f"g_value deve essere > 0; valore={g}", fixed=False)

            # dimensioni finestra
            h = s.get("height", None)
            w = s.get("width", None)
            if not (isinstance(h, (int, float)) and h > 0):
                if fix:
                    s["height"] = 1.0
                    add_issue("WARN", f"{path}.height", f"height non valido ({h}); impostato a 1.0", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.height", f"height deve essere > 0; valore={h}", fixed=False)
            if not (isinstance(w, (int, float)) and w > 0):
                if fix:
                    s["width"] = 1.0
                    add_issue("WARN", f"{path}.width", f"width non valido ({w}); impostato a 1.0", fixed=True)
                else:
                    add_issue("ERROR", f"{path}.width", f"width deve essere > 0; valore={w}", fixed=False)

            # parapet 0..height
            para = s.get("parapet", None)
            if para is not None and isinstance(para, (int, float)) and h is not None and isinstance(h, (int,float)):
                if not (0.0 <= para <= max(h, 0.0)):
                    if fix:
                        s["parapet"] = min(max(0.0, para if isinstance(para, (int,float)) else 0.0), h if isinstance(h,(int,float)) else 1.0)
                        add_issue("WARN", f"{path}.parapet", f"parapet fuori range; clippato a {s['parapet']}", fixed=True)
                    else:
                        add_issue("WARN", f"{path}.parapet", f"parapet fuori range (0..height); valore={para}, height={h}", fixed=False)

        # orientamenti consigliati
        ori = s.get("orientation", {})
        tilt = ori.get("tilt", None)
        az   = ori.get("azimuth", None)
        if not (isinstance(tilt, (int, float)) and tilt in (0, 90)):
            if fix and isinstance(tilt, (int, float)):
                s["orientation"]["tilt"] = 0 if abs(tilt - 0) < abs(tilt - 90) else 90
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt non standard ({tilt}); normalizzato a {s['orientation']['tilt']}", fixed=True)
            else:
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt consigliato 0 o 90; valore={tilt}", fixed=False)
        if not (isinstance(az, (int, float)) and az in (0, 90, 180, 270)):
            if fix and isinstance(az, (int, float)):
                snapped = int(round(az / 90.0) * 90) % 360
                if snapped == 360: snapped = 0
                s["orientation"]["azimuth"] = snapped
                add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth non standard ({az}); normalizzato a {snapped}", fixed=True)
            else:
                add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth consigliato in {{0,90,180,270}}; valore={az}", fixed=False)

    # ---------------------------
    # 3) ADJACENT ZONES CHECKS
    # ---------------------------
    for j, az in enumerate(bui_clean.get("adjacent_zones", [])):
        path = f"adjacent_zones[{j}]"
        vol = az.get("volume", None)
        if not (isinstance(vol, (int, float)) and vol > 0):
            if fix and isinstance(vol, (int, float)) and vol == 0:
                az["volume"] = 1.0
                add_issue("WARN", f"{path}.volume", "volume era 0; impostato a 1.0", fixed=True)
            else:
                add_issue("ERROR", f"{path}.volume", f"volume deve essere > 0; valore={vol}", fixed=False)

        # coerenza lunghezze
        arrs = {
            "area_facade_elements": az.get("area_facade_elements"),
            "typology_elements": az.get("typology_elements"),
            "transmittance_U_elements": az.get("transmittance_U_elements"),
            "orientation_elements": az.get("orientation_elements"),
        }
        lengths = [len(v) for v in arrs.values() if isinstance(v, (list, np.ndarray))]
        if lengths and len(set(lengths)) != 1:
            add_issue("ERROR", path, f"Lunghezze arrays non coerenti: {[ (k, None if v is None else len(v)) for k,v in arrs.items() ]}", fixed=False)

        # U elementi > 0
        U = arrs.get("transmittance_U_elements")
        if isinstance(U, (list, np.ndarray)):
            U = np.asarray(U, dtype=float)
            bad = np.where(U <= 0)[0].tolist()
            if bad:
                if fix:
                    U[U <= 0] = D["opaque_u_default"]
                    az["transmittance_U_elements"] = U
                    add_issue("WARN", f"{path}.transmittance_U_elements", f"U<=0 ai idx {bad}; impostati a {D['opaque_u_default']}", fixed=True)
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

        # Consideriamo solo pareti verticali
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
            add_issue("ERROR", path, f"name_adj_zone='{adj_name}' non trovata in adjacent_zones", fixed=False)
            i += 1
            continue

        # --- arrays zona adiacente (writable) ---
        azone = adj_by_name[adj_name]
        A_arr  = np.array(azone.get("area_facade_elements", []), dtype=float).copy()
        TypArr = np.array(azone.get("typology_elements", []),     dtype=object)
        OriArr = np.array(azone.get("orientation_elements", []),  dtype=object)

        if not (len(A_arr) == len(TypArr) == len(OriArr)):
            add_issue("ERROR", f"adjacent_zones[{adj_name}]", "Arrays non coerenti (len mismatch) per il cap per direzione", fixed=False)
            i += 1
            continue

        # === (1) CAP area OP della zona adiacente sulla direzione ===
        mask_op_dir = (np.char.upper(TypArr.astype(str)) == "OP") & (np.char.upper(OriArr.astype(str)) == dir_card)
        idx = np.where(mask_op_dir)[0]
        if idx.size == 0:
            A_adj_dir_cap = 0.0
        else:
            A_adj_dir_cap = float(A_arr[idx].sum())
            if A_adj_dir_cap > A_bui_orig + eps:
                msg = (f"Zona '{adj_name}' OP {dir_card}={A_adj_dir_cap:.3f} m² > parete BUI {A_bui_orig:.3f} m² "
                       f"({s.get('name','(senza nome)')})")
                if not fix:
                    add_issue("ERROR", path, msg + " (no fix)", fixed=False)
                else:
                    sf = A_bui_orig / max(A_adj_dir_cap, eps)
                    A_arr[idx] = A_arr[idx] * sf
                    azone["area_facade_elements"] = A_arr  # write-back
                    A_adj_dir_cap = float(A_arr[idx].sum())
                    add_issue("WARN", path, msg + f" → ridotto (scale {sf:.3f})", fixed=True)

        # === (2) WINDOW CHECK nella stessa direzione (usa A_bui_orig e A_adj_dir_cap) ===
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
                msg = (f"Finestre {dir_card}: area finestre={A_win_dir:.3f} m² > differenza "
                       f"(A_BUI_adjacent - A_adj_OP)={diff:.3f} m² su {s.get('name','(senza nome)')}")
                if not fix:
                    add_issue("ERROR", path, msg + " (no fix)", fixed=False)
                else:
                    # scala proporzionalmente tutte le finestre in questa direzione per farle sommare a 'diff'
                    sfw = (diff / max(A_win_dir, eps)) if diff > 0 else 0.0
                    for k in win_idx:
                        oldA = float(bui_clean["building_surface"][k].get("area", 0.0))
                        newA = oldA * sfw
                        bui_clean["building_surface"][k]["area"] = newA
                    add_issue("WARN", path, msg + f" → aree finestre ridotte (scale {sfw:.3f})", fixed=True)

        # === (3) SPLIT: se la zona adiacente copre meno di A_bui_orig, crea nuova parete 'opaque' per il residuo ===
        diff_ext = max(A_bui_orig - A_adj_dir_cap, 0.0)
        if diff_ext > eps:
            if not fix:
                add_issue("ERROR", path, (f"Parete 'adjacent' {s.get('name','(senza nome)')} non coperta interamente dalla "
                                          f"zona '{adj_name}' ({A_adj_dir_cap:.3f} < {A_bui_orig:.3f}). "
                                          f"Residuo {diff_ext:.3f} m² non allocato."), fixed=False)
            else:
                # crea nuovo elemento OPAQUE per il residuo
                new_surf = {
                    "name": f"{s.get('name','(adjacent)')} — residuo esterno",
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

                # ridimensiona la parete adjacent alla sola area coperta dalla zona adiacente
                s["area"] = float(A_adj_dir_cap)

                add_issue("WARN", path, (f"Creato elemento 'opaque' residuo di {diff_ext:.3f} m² e "
                                         f"ridimensionata parete 'adjacent' a {A_adj_dir_cap:.3f} m²."), fixed=True)

        # passo al prossimo elemento (attenzione: la lista può essere cresciuta)
        i += 1

        # ----------------------------------------------------------
        # 4) SPLIT PARETI ADJACENT: parte adiacente + parte esterna
        #    Crea nuova superficie "opaque" per la porzione esterna.
        #    (ESEGUIRE DOPO TUTTI I CHECK)
        # ----------------------------------------------------------
        new_surfaces = []   # colleziono qui e aggiungo con .extend a fine loop
        for i, s in enumerate(bui_clean.get("building_surface", [])):
            if (s.get("type") != "adjacent") or (not s.get("name_adj_zone")):
                continue

            # Direzione NV/EV/SV/WV/HOR della superficie BUI
            az = s.get("orientation", {}).get("azimuth", 0.0)
            tl = s.get("orientation", {}).get("tilt", 90.0)
            dir_code = _dir_from_orientation(az, tl)

            # Consideriamo SOLO le verticali per lo split con zona adiacente
            if dir_code not in ("NV","EV","SV","WV"):
                continue

            A_bui = float(s.get("area", 0.0))
            if A_bui <= 0:
                continue

            # Trova la zona adiacente corrispondente per nome
            adj_name = s.get("name_adj_zone")
            adj_zone = None
            for azj in bui_clean.get("adjacent_zones", []):
                if azj.get("name") == adj_name:
                    adj_zone = azj
                    break
            if adj_zone is None:
                # Nome non trovato: segnalo
                add_issue("WARN", f"building_surface[{i}]", f'name_adj_zone="{adj_name}" non trovato tra le adjacent_zones', fixed=False)
                continue

            # Area OP della adj-zone nella stessa direzione
            A_adj_dir = _adj_op_area_in_dir(adj_zone, dir_code)

            # CAP: l’adiacente non può eccedere la parete BUI
            if A_adj_dir > A_bui:
                add_issue("WARN",
                        f"building_surface[{i}]",
                        f"Area zona adiacente {adj_name} in {dir_code} ({A_adj_dir:.3f} m²) > area BUI ({A_bui:.3f} m²); cappata a {A_bui:.3f}",
                        fixed=True)
                A_adj_dir = A_bui

            # Se la zona adiacente è PIÙ PICCOLA della parete BUI → crea residuo esterno
            diff = A_bui - A_adj_dir
            if diff > 1e-9:
                # 1) riduci la superficie ADJACENT alla sola parte a contatto con la zona non termica
                old_area = s["area"]
                s["area"] = A_adj_dir

                # 2) crea una NUOVA superficie "opaque" esterna con area = diff
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
                    "name_adj_zone": None,   # importantissimo: esterno, non adiacente
                }

                # se il coeff. conv./rad. sono già stati idratati, copiali (non obbligatorio)
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
                        f"Split {dir_code}: ridotta area ADJ da {old_area:.3f} a {A_adj_dir:.3f} m²; creata nuova OPAQUE esterna da {diff:.3f} m².",
                        fixed=True)

        # Aggiungo in coda TUTTE le nuove superfici
        if new_surfaces:
            bui_clean["building_surface"].extend(new_surfaces)


    return bui_clean, issues


    


# def sanitize_and_validate_BUI(bui: dict,
#                               fix: bool = False,
#                               eps: float = 1e-6,
#                               defaults: dict | None = None):
#     """
#     Validate and (optionally) sanitize a BUI.
    
#     Parameters
#     ----------
#     bui : dict
#         The BUI dictionary to validate.
#     fix : bool
#         If True, attempts to correct invalid/zero values (clipping to eps, defaults, etc.).
#     eps : float
#         Minimum value for clipping of positive quantities.
#     defaults : dict | None
#         Optional physical defaults, e.g.:
#         {
#           "opaque_u_default": 0.5,
#           "transparent_u_default": 1.6,
#           "g_default": 0.6
#         }
    
#     Returns
#     -------
#     bui_clean : dict
#         Copy of the BUI (modified if fix=True).
#     issues : list[dict]
#         List of issues found: {"level": "ERROR|WARN", "path": str, "msg": str, "fix_applied": bool}
#     """

#     bui_clean = deepcopy(bui)
#     issues = []
#     D = defaults or {
#         "opaque_u_default": 0.5,
#         "transparent_u_default": 1.6,
#         "g_default": 0.6
#     }

#     def add_issue(level, path, msg, fixed=False):
#         issues.append({"level": level, "path": path, "msg": msg, "fix_applied": fixed})

#     # ---------------------------
#     # 1) BUILDING-LEVEL CHECKS
#     # ---------------------------
#     b = bui_clean.get("building", {})
#     for key in ["net_floor_area", "exposed_perimeter", "height"]:
#         if key in b:
#             val = b[key]
#             if not (isinstance(val, (int, float)) and val > 0):
#                 if fix and isinstance(val, (int, float)) and val == 0:
#                     b[key] = max(eps, 1.0)
#                     add_issue("WARN", f"building.{key}", f"{key} era 0; impostato a {b[key]}", fixed=True)
#                 else:
#                     add_issue("ERROR", f"building.{key}", f"{key} deve essere > 0; valore={val}", fixed=False)

#     if "n_floors" in b and (not isinstance(b["n_floors"], (int, float)) or b["n_floors"] <= 0):
#         if fix and isinstance(b["n_floors"], (int, float)) and b["n_floors"] == 0:
#             b["n_floors"] = 1
#             add_issue("WARN", "building.n_floors", "n_floors era 0; impostato a 1", fixed=True)
#         else:
#             add_issue("ERROR", "building.n_floors", f"n_floors deve essere > 0; valore={b.get('n_floors')}", fixed=False)

#     # ---------------------------
#     # 2) SURFACES CHECKS
#     # ---------------------------
#     type_corrections = {"opque": "opaque", "opaqu": "opaque", "trasparent": "transparent"}
#     allowed_types = {"opaque", "transparent", "adiabatic", "adjacent"}

#     for i, s in enumerate(bui_clean.get("building_surface", [])):
#         path = f"building_surface[{i}]"
#         # Correggi/refusi type
#         t = s.get("type")
#         if isinstance(t, str) and t.lower() in type_corrections:
#             old = t
#             s["type"] = type_corrections[t.lower()]
#             add_issue("WARN", f"{path}.type", f'type="{old}" corretto in "{s["type"]}"', fixed=True)
#         t = (s.get("type") or "").lower()
#         if t not in allowed_types:
#             add_issue("ERROR", f"{path}.type", f'type "{s.get("type")}" non riconosciuto', fixed=False)

#         # area
#         area = s.get("area", None)
#         if not (isinstance(area, (int, float)) and area > 0):
#             if fix and isinstance(area, (int, float)) and area == 0:
#                 s["area"] = max(eps, 1.0)
#                 add_issue("WARN", f"{path}.area", f"area was 0; set to {s['area']}", fixed=True)
#             else:
#                 add_issue("ERROR", f"{path}.area", f"area must be > 0; value={area}", fixed=False)

#         # sky_view_factor
#         svf = s.get("sky_view_factor", None)
#         if not (isinstance(svf, (int, float)) and 0.0 <= svf <= 1.0):
#             if fix and isinstance(svf, (int, float)):
#                 s["sky_view_factor"] = min(1.0, max(0.0, svf))
#                 add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor out of range; clippato a {s['sky_view_factor']}", fixed=True)
#             else:
#                 add_issue("WARN", f"{path}.sky_view_factor", f"sky_view_factor missing or not in [0,1]; value={svf}", fixed=False)

#         # U-value per opaque/transparent
#         if t in {"opaque", "transparent"}:
#             u = s.get("u_value", None)
#             if not (isinstance(u, (int, float)) and u > 0):
#                 if fix:
#                     default_u = D["opaque_u_default"] if t == "opaque" else D["transparent_u_default"]
#                     s["u_value"] = default_u
#                     add_issue("WARN", f"{path}.u_value", f"u_value was invalid ({u}); set to default {default_u}", fixed=True)
#                 else:
#                     add_issue("ERROR", f"{path}.u_value", f"u_value must be > 0; value={u}", fixed=False)

#         # g-value per trasparenti
#         if t == "transparent":
#             g = s.get("g_value", None)
#             if not (isinstance(g, (int, float)) and g > 0):
#                 if fix:
#                     s["g_value"] = D["g_default"]
#                     add_issue("WARN", f"{path}.g_value", f"g_value was invalid ({g}); set to default {D['g_default']}", fixed=True)
#                 else:
#                     add_issue("ERROR", f"{path}.g_value", f"g_value must be > 0; value={g}", fixed=False)

#             # dimensioni finestra
#             h = s.get("height", None)
#             w = s.get("width", None)
#             if not (isinstance(h, (int, float)) and h > 0):
#                 if fix:
#                     s["height"] = 1.0
#                     add_issue("WARN", f"{path}.height", f"height was invalid ({h}); set to 1.0", fixed=True)
#                 else:
#                     add_issue("ERROR", f"{path}.height", f"height must be > 0; value={h}", fixed=False)
#             if not (isinstance(w, (int, float)) and w > 0):
#                 if fix:
#                     s["width"] = 1.0
#                     add_issue("WARN", f"{path}.width", f"width was invalid ({w}); set to 1.0", fixed=True)
#                 else:
#                     add_issue("ERROR", f"{path}.width", f"width must be > 0; value={w}", fixed=False)

#             # parapet 0..height
#             para = s.get("parapet", None)
#             if para is not None and isinstance(para, (int, float)) and h is not None and isinstance(h, (int,float)):
#                 if not (0.0 <= para <= max(h, 0.0)):
#                     if fix:
#                         s["parapet"] = min(max(0.0, para if isinstance(para, (int,float)) else 0.0), h if isinstance(h,(int,float)) else 1.0)
#                         add_issue("WARN", f"{path}.parapet", f"parapet was out of range; clippato a {s['parapet']}", fixed=True)
#                     else:
#                         add_issue("WARN", f"{path}.parapet", f"parapet was out of range (0..height); value={para}, height={h}", fixed=False)

#         # orientamenti
#         ori = s.get("orientation", {})
#         tilt = ori.get("tilt", None)
#         az   = ori.get("azimuth", None)

#         # tilt 0 o 90 consigliati
#         if not (isinstance(tilt, (int, float)) and tilt in (0, 90)):
#             if fix and isinstance(tilt, (int, float)):
#                 s["orientation"]["tilt"] = 0 if abs(tilt - 0) < abs(tilt - 90) else 90
#                 add_issue("WARN", f"{path}.orientation.tilt", f"tilt was non standard ({tilt}); normalized to {s['orientation']['tilt']}", fixed=True)
#             else:
#                 add_issue("WARN", f"{path}.orientation.tilt", f"tilt was non standard ({tilt}); normalized to {s['orientation']['tilt']}", fixed=False)

#         # azimuth 0/90/180/270 consigliati
#         if not (isinstance(az, (int, float)) and az in (0, 90, 180, 270)):
#             if fix and isinstance(az, (int, float)):
#                 snapped = int(round(az / 90.0) * 90) % 360
#                 if snapped == 360: snapped = 0
#                 s["orientation"]["azimuth"] = snapped
#                 add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth non standard ({az}); normalizzato a {snapped}", fixed=True)
#             else:
#                 add_issue("WARN", f"{path}.orientation.azimuth", f"azimuth consigliato in {{0,90,180,270}}; valore={az}", fixed=False)

#     # ---------------------------
#     # 3) ADJACENT ZONES CHECKS
#     # ---------------------------
#     for j, az in enumerate(bui_clean.get("adjacent_zones", [])):
#         path = f"adjacent_zones[{j}]"
#         vol = az.get("volume", None)
#         if not (isinstance(vol, (int, float)) and vol > 0):
#             if fix and isinstance(vol, (int, float)) and vol == 0:
#                 az["volume"] = 1.0
#                 add_issue("WARN", f"{path}.volume", "volume era 0; impostato a 1.0", fixed=True)
#             else:
#                 add_issue("ERROR", f"{path}.volume", f"volume deve essere > 0; valore={vol}", fixed=False)

#         # coerenza lunghezze
#         arrs = {
#             "area_facade_elements": az.get("area_facade_elements"),
#             "typology_elements": az.get("typology_elements"),
#             "transmittance_U_elements": az.get("transmittance_U_elements"),
#             "orientation_elements": az.get("orientation_elements"),
#         }
#         lengths = [len(v) for v in arrs.values() if isinstance(v, (list, np.ndarray))]
#         if lengths and len(set(lengths)) != 1:
#             add_issue("ERROR", path, f"Lengths of arrays are not coherent: {[ (k, None if v is None else len(v)) for k,v in arrs.items() ]}", fixed=False)

#         # U elementi > 0
#         U = arrs.get("transmittance_U_elements")
#         if isinstance(U, (list, np.ndarray)):
#             U = np.asarray(U, dtype=float)
#             bad = np.where(U <= 0)[0].tolist()
#             if bad:
#                 if fix:
#                     U[U <= 0] = D["opaque_u_default"]
#                     az["transmittance_U_elements"] = U
#                     add_issue("WARN", f"{path}.transmittance_U_elements", f"U<=0 ai idx {bad}; impostati a {D['opaque_u_default']}", fixed=True)
#                 else:
#                     add_issue("ERROR", f"{path}.transmittance_U_elements", f"U<=0 ai idx {bad}", fixed=False)

#     # -------------------------------------------------------------------
#     # 4) ADJACENCY PAIRING CAPS: OP (verticale) della zona ≤ area BUI
#     # -------------------------------------------------------------------
#     def _azimuth_to_dir(az: float) -> str | None:
#         """Mappa 0/90/180/270 -> 'NV'/'EV'/'SV'/'WV'. Altri valori -> None."""
#         if az is None:
#             return None
#         azn = int(round(float(az))) % 360
#         if azn in (0, 360): return "NV"
#         if azn == 90:       return "EV"
#         if azn == 180:      return "SV"
#         if azn == 270:      return "WV"
#         return None

#     # Mappa nome zona -> dizionario zona (per lookup rapido)
#     adj_by_name = { z.get("name"): z for z in bui_clean.get("adjacent_zones", []) }

#     for i, s in enumerate(bui_clean.get("building_surface", [])):
#         path = f"building_surface[{i}]"
#         if (s.get("type") != "adjacent") or (not s.get("name_adj_zone")):
#             continue

#         az = s.get("orientation", {}).get("azimuth", None)
#         tilt = s.get("orientation", {}).get("tilt", None)

#         # Consideriamo solo pareti verticali del BUI (tilt≈90) con direzione cardinale nota
#         if not (isinstance(tilt, (int,float)) and abs(tilt-90) < 1e-6):
#             continue

#         dir_card = _azimuth_to_dir(az)
#         if dir_card is None:  # orizzontali o azimut non standard -> skip
#             continue

#         A_bui = s.get("area", None)
#         if not (isinstance(A_bui, (int,float)) and A_bui > 0):
#             continue  # già segnalato sopra

#         adj_name = s["name_adj_zone"]
#         if adj_name not in adj_by_name:
#             add_issue("ERROR", path, f"name_adj_zone='{adj_name}' not found in adjacent_zones", fixed=False)
#             continue

#         azone = adj_by_name[adj_name]

#         # Assicuriamoci che gli array siano numpy *scrivibili*
#         A_arr  = np.array(azone.get("area_facade_elements", []), dtype=float).copy()
#         TypArr = np.array(azone.get("typology_elements", []),     dtype=object)
#         OriArr = np.array(azone.get("orientation_elements", []),  dtype=object)

#         if not (len(A_arr) == len(TypArr) == len(OriArr)):
#             add_issue("ERROR", f"adjacent_zones[{adj_name}]", "Arrays are not coherent (len mismatch) for the cap for direction", fixed=False)
#             continue

#         # Indici: OP and direction equal to dir_card (NV/EV/SV/WV)
#         mask = (np.char.upper(TypArr.astype(str)) == "OP") & (np.char.upper(OriArr.astype(str)) == dir_card)
#         idx = np.where(mask)[0]
#         if idx.size == 0:
#             # Nessuna parete OP in quella direzione: nulla da fare
#             continue

#         A_adj_dir = float(A_arr[idx].sum())
#         if A_adj_dir <= A_bui + 1e-9:
#             # OK: già ≤ BUI
#             continue

#         # Serve riduzione: o errore (fix=False) o correzione (fix=True)
#         msg = (f"Zona '{adj_name}' has OP area {dir_card}={A_adj_dir:.3f} m² superiore "
#                f"all’area BUI {A_bui:.3f} m² associated to {s.get('name','(senza nome)')}")
#         if not fix:
#             add_issue("ERROR", path, msg + " (no fix applied)", fixed=False)
#             continue

#         # Proportional reduction
#         sf = A_bui / max(A_adj_dir, eps)
#         A_arr[idx] = A_arr[idx] * sf

#         # Scriviamo indietro gli array *writable*
#         azone["area_facade_elements"] = A_arr
#         add_issue("WARN", path, msg + f" → redistributed proportionally (scale {sf:.3f})", fixed=True)

#     return bui_clean, issues
