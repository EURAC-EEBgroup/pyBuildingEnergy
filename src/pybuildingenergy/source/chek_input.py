from copy import deepcopy
import numpy as np

def sanitize_and_validate_BUI(bui: dict,
                              fix: bool = False,
                              eps: float = 1e-6,
                              defaults: dict | None = None):
    """
    Valida e (opzionalmente) sanifica un BUI.
    
    Parametri
    ---------
    bui : dict
        Il dizionario BUI da controllare.
    fix : bool
        Se True prova a correggere valori non validi/zero (clip a eps, default, ecc.).
    eps : float
        Valore minimo per clipping di grandezze >0.
    defaults : dict | None
        Default fisici opzionali, es.:
        {
          "opaque_u_default": 0.5,
          "transparent_u_default": 1.6,
          "g_default": 0.6
        }
    
    Ritorna
    -------
    bui_clean : dict
        Copia del BUI (modificata se fix=True).
    issues : list[dict]
        Elenco di problemi trovati: {"level": "ERROR|WARN", "path": str, "msg": str, "fix_applied": bool}
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
        # Correggi/refusi type
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

        # U-value: deve essere >0 per opaque e transparent (non serve per adiabatic/adjacent)
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

        # orientamenti
        ori = s.get("orientation", {})
        tilt = ori.get("tilt", None)
        az   = ori.get("azimuth", None)

        # tilt 0 or 90 (se vuoi accettare qualunque angolo 0..180, commenta questo blocco e abilita il check di range)
        if not (isinstance(tilt, (int, float)) and tilt in (0, 90)):
            if fix and isinstance(tilt, (int, float)):
                # porta a 0 o 90 il più vicino
                s["orientation"]["tilt"] = 0 if abs(tilt - 0) < abs(tilt - 90) else 90
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt non standard ({tilt}); normalizzato a {s['orientation']['tilt']}", fixed=True)
            else:
                add_issue("WARN", f"{path}.orientation.tilt", f"tilt consigliato 0 o 90; valore={tilt}", fixed=False)

        # azimuth 0/90/180/270 (idem come sopra)
        if not (isinstance(az, (int, float)) and az in (0, 90, 180, 270)):
            if fix and isinstance(az, (int, float)):
                # snap al più vicino multiplo di 90
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

        # coerenza lunghezze array principali, se presenti
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

    return bui_clean, issues
