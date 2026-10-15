"""3D view of the BESTEST test cells for the HTML reports of the ASHRAE
140-2023 (Section 7) and EN ISO 52016-1:2017 (clause 7.2) harnesses.

Geometry from Figures 7-1 (Case 600), 7-3 (610 overhang), 7-4 (620 east/west
windows), 7-5/7-6 (630 overhang and fins) and 7-9 (960 sunspace); the ISO
52016-1 cases 600/640/600FF and 900/940/900FF share the Case 600/900 cell.
The view shows what distinguishes each case: window orientation, shading
devices, glazing type (660 low-e, 670 single pane), increased wall and roof
insulation (680/685/695, 980/985/995), the night-ventilation fan (650/950)
and the sunspace (960). Thermostat strategies have no geometry and are named
in the caption. Axes: x = east, y = north, z = up, origin at the south-west
corner of the floor.
"""

from __future__ import annotations

import plotly.graph_objects as go

L, W, H = 8.0, 6.0, 2.7          # building length (E-W), depth (N-S), height [m]
SILL, WIN_H, WIN_W = 0.2, 2.0, 3.0
LIFT = 0.25                      # raised floor drawn above the ground (Section 7.2.1.5)
EPS = 0.015                      # offset of windows / slabs from the wall plane
SCALE = 0.2                      # scene units per metre (size of the building in the frame)

COLORS = {
    "light_wall": "#d9c7a7", "heavy_wall": "#a9a39a", "roof": "#8c6d5a", "floor": "#b9b2a6",
    "window": "#5aa9e6", "shading": "#5b6670", "common_wall": "#7d7468", "sun_wall": "#a9a39a",
    "edge": "#2b2f33", "ground": "#e7ece4", "insulation": "#f2d16b", "airflow": "#1f7a4d",
    "window_lowe": "#3fbfa8", "window_single": "#b9dff7",
}

WINDOW_STYLE = {   # glazing type -> (legend label, colour, opacity)
    "double": ("Windows (double clear, Table 7-10)", COLORS["window"], 0.75),
    "lowe": ("Windows (low-e, argon-filled, Table 7-17)", COLORS["window_lowe"], 0.75),
    "single": ("Windows (single pane, Table 7-21)", COLORS["window_single"], 0.6),
}
INSULATION_M = {"wall_light": 0.25, "wall_heavy": 0.2452, "roof": 0.4}   # Tables 7-25 and 7-31


def _source_case(case_number: int) -> int:
    """600-series perturbation a case is built on (as build_bestest_case)."""
    if case_number in (980, 985, 995):
        return {980: 680, 985: 685, 995: 695}[case_number]
    return case_number - 300 if case_number >= 900 else case_number


class _Mesh:
    """Collect quads per category; one Mesh3d trace per category."""

    def __init__(self):
        self.quads: dict[str, list] = {}

    def quad(self, cat, p0, p1, p2, p3):
        self.quads.setdefault(cat, []).append((p0, p1, p2, p3))

    def box(self, cat, x0, x1, y0, y1, z0, z1):
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for a, b, c, d in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.quad(cat, p[a], p[b], p[c], p[d])

    def traces(self, styles):
        out, ex, ey, ez = [], [], [], []
        for cat, quads in self.quads.items():
            label, color, opacity = styles[cat]
            xs, ys, zs, i, j, k = [], [], [], [], [], []
            for q in quads:
                n = len(xs)
                for x, y, z in q:
                    xs.append(x), ys.append(y), zs.append(z)
                i += [n, n]
                j += [n + 1, n + 2]
                k += [n + 2, n + 3]
                if cat != "ground":
                    for x, y, z in list(q) + [q[0]]:
                        ex.append(x), ey.append(y), ez.append(z)
                    ex.append(None), ey.append(None), ez.append(None)
            out.append(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, name=label, color=color,
                                 opacity=opacity, flatshading=True, showlegend=cat != "ground",
                                 hoverinfo="name", lighting=dict(ambient=0.92, diffuse=0.25, specular=0.0, fresnel=0.0)))
        out.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color=COLORS["edge"], width=2),
                                hoverinfo="skip", showlegend=False))
        return out


def _wall_with_openings(m, cat, plane, fixed, span, openings):
    """Opaque wall on plane 'y' (south/north) or 'x' (east/west), split around
    rectangular openings [(u0, u1, z0, z1)] so windows are true holes."""
    u0, u1 = span
    cuts_u = sorted({u0, u1, *[o[0] for o in openings], *[o[1] for o in openings]})
    cuts_z = sorted({LIFT, LIFT + H, *[o[2] for o in openings], *[o[3] for o in openings]})
    for a, b in zip(cuts_u, cuts_u[1:]):
        for c, d in zip(cuts_z, cuts_z[1:]):
            inside = any(o[0] <= a and b <= o[1] and o[2] <= c and d <= o[3] for o in openings)
            if inside:
                continue
            if plane == "y":
                m.quad(cat, (a, fixed, c), (b, fixed, c), (b, fixed, d), (a, fixed, d))
            else:
                m.quad(cat, (fixed, a, c), (fixed, b, c), (fixed, b, d), (fixed, a, d))


def _window(m, plane, fixed, u0, u1, z0, z1, outward):
    f = fixed + outward * EPS
    if plane == "y":
        m.quad("window", (u0, f, z0), (u1, f, z0), (u1, f, z1), (u0, f, z1))
    else:
        m.quad("window", (f, u0, z0), (f, u1, z0), (f, u1, z1), (f, u0, z1))


def _insulation_layer(m, plane, fixed, span, openings, t, outward, z_lo, z_hi):
    """Extruded exterior insulation, leaving the window openings (reveals)."""
    u0, u1 = span
    cuts_u = sorted({u0, u1, *[o[0] for o in openings], *[o[1] for o in openings]})
    cuts_z = sorted({z_lo, z_hi, *[o[2] for o in openings], *[o[3] for o in openings]})
    for a, b in zip(cuts_u, cuts_u[1:]):
        for c, d in zip(cuts_z, cuts_z[1:]):
            if any(o[0] <= a and b <= o[1] and o[2] <= c and d <= o[3] for o in openings):
                continue
            n0, n1 = sorted((fixed + outward * EPS, fixed + outward * t))
            if plane == "y":
                m.box("insulation", a, b, n0, n1, c, d)
            else:
                m.box("insulation", n0, n1, a, b, c, d)


def building_figure(case_number: int, heavy: bool, free_float: bool = False) -> go.Figure:
    src = 600 if case_number == 960 else _source_case(case_number)
    glazing = "lowe" if src == 660 else "single" if src == 670 else "double"
    more_insulation = src in (680, 695)   # also 980/995: Tables 7-25 and 7-31
    night_vent = src == 650
    m = _Mesh()
    z0, z_top = LIFT, LIFT + H
    wall = "heavy_wall" if heavy else "light_wall"
    pad = 1.6

    if case_number == 960:
        # Sun zone (y 0-2, high mass) in front of the back zone (y 2-8, low mass), Figure 7-9.
        d = 2.0
        sun_wins = [(0.5, 3.5, z0 + SILL + 0.3, z0 + SILL + 0.3 + WIN_H),
                    (4.5, 7.5, z0 + SILL + 0.3, z0 + SILL + 0.3 + WIN_H)]
        _wall_with_openings(m, "sun_wall", "y", 0.0, (0, L), sun_wins)
        for a, b, c, dd in sun_wins:
            _window(m, "y", 0.0, a, b, c, dd, -1)
        for x in (0.0, L):
            m.quad("sun_wall", (x, 0, z0), (x, d, z0), (x, d, z_top), (x, 0, z_top))
        m.quad("roof", (0, 0, z_top), (L, 0, z_top), (L, d, z_top), (0, d, z_top))
        m.quad("floor", (0, 0, z0), (L, 0, z0), (L, d, z0), (0, d, z0))
        m.quad("common_wall", (0, d, z0), (L, d, z0), (L, d, z_top), (0, d, z_top))
        y0, y1 = d, d + W
        m.quad("light_wall", (0, y1, z0), (L, y1, z0), (L, y1, z_top), (0, y1, z_top))
        for x in (0.0, L):
            m.quad("light_wall", (x, y0, z0), (x, y1, z0), (x, y1, z_top), (x, y0, z_top))
        m.quad("roof", (0, y0, z_top), (L, y0, z_top), (L, y1, z_top), (0, y1, z_top))
        m.quad("floor", (0, y0, z0), (L, y0, z0), (L, y1, z0), (0, y1, z0))
        depth = d + W
    else:
        south, east, west = [], [], []
        if src in (620, 630):
            win = (1.5, 4.5, z0 + SILL, z0 + SILL + WIN_H)
            east, west = [win], [win]
        else:
            south = [(0.5, 3.5, z0 + SILL, z0 + SILL + WIN_H), (4.5, 7.5, z0 + SILL, z0 + SILL + WIN_H)]
        _wall_with_openings(m, wall, "y", 0.0, (0, L), south)
        _wall_with_openings(m, wall, "y", W, (0, L), [])
        _wall_with_openings(m, wall, "x", L, (0, W), east)
        _wall_with_openings(m, wall, "x", 0.0, (0, W), west)
        for a, b, c, d in south:
            _window(m, "y", 0.0, a, b, c, d, -1)
        for a, b, c, d in east:
            _window(m, "x", L, a, b, c, d, +1)
        for a, b, c, d in west:
            _window(m, "x", 0.0, a, b, c, d, -1)
        m.quad("roof", (0, 0, z_top), (L, 0, z_top), (L, W, z_top), (0, W, z_top))
        m.quad("floor", (0, 0, z0), (L, 0, z0), (L, W, z0), (0, W, z0))
        t = 0.04
        if src == 610:  # Figure 7-3: 1.0 m overhang at roof level across the south wall
            m.box("shading", 0, L, -1.0, -EPS, z_top - t, z_top)
        if src == 630:  # Figures 7-5/7-6: window-wide overhang + full-height fins, 1.0 m deep
            for x_wall, sgn in ((L, +1), (0.0, -1)):
                xa, xb = sorted((x_wall + sgn * EPS, x_wall + sgn * 1.0))
                m.box("shading", xa, xb, 1.5, 4.5, z_top - t, z_top)
                for y_f in (1.5, 4.5):
                    m.box("shading", xa, xb, y_f - t / 2, y_f + t / 2, 0.0, z_top)
        depth = W

        if night_vent:  # inlet on the south-wall pier, outlet on the east wall (schematic)
            zc = z0 + H / 2
            m.quad("vent", (3.7, -EPS, zc - 0.3), (4.3, -EPS, zc - 0.3), (4.3, -EPS, zc + 0.3), (3.7, -EPS, zc + 0.3))
            m.quad("vent", (L + EPS, W / 2 - 0.3, zc - 0.3), (L + EPS, W / 2 + 0.3, zc - 0.3),
                   (L + EPS, W / 2 + 0.3, zc + 0.3), (L + EPS, W / 2 - 0.3, zc + 0.3))

        if more_insulation:  # drawn at its real thickness, outside the structure
            t_wall = INSULATION_M["wall_heavy" if heavy else "wall_light"]
            _insulation_layer(m, "y", 0.0, (0, L), south, t_wall, -1, z0, z_top)
            _insulation_layer(m, "y", W, (0, L), [], t_wall, +1, z0, z_top)
            _insulation_layer(m, "x", L, (0, W), east, t_wall, +1, z0, z_top)
            _insulation_layer(m, "x", 0.0, (0, W), west, t_wall, -1, z0, z_top)
            m.box("insulation", 0, L, 0, W, z_top + EPS, z_top + INSULATION_M["roof"])

    m.quad("ground", (-pad, -pad - 0.4, 0), (L + pad, -pad - 0.4, 0), (L + pad, depth + pad, 0), (-pad, depth + pad, 0))

    styles = {
        "light_wall": ("Walls (low mass)", COLORS["light_wall"], 1.0),
        "heavy_wall": ("Walls (high mass)", COLORS["heavy_wall"], 1.0),
        "sun_wall": ("Sun-zone walls (high mass)", COLORS["sun_wall"], 1.0),
        "common_wall": ("Common wall (0.20 m concrete)", COLORS["common_wall"], 1.0),
        "roof": ("Roof", COLORS["roof"], 1.0),
        "floor": ("Raised floor", COLORS["floor"], 1.0),
        "window": WINDOW_STYLE[glazing],
        "insulation": ("Increased insulation (Tables 7-25 / 7-31)", COLORS["insulation"], 1.0),
        "shading": ("Overhang / fins", COLORS["shading"], 0.55),
        "ground": ("Ground", COLORS["ground"], 1.0),
        "vent": ("Vent fan openings (schematic)", COLORS["airflow"], 1.0),
    }
    fig = go.Figure(m.traces(styles))

    # North arrow, orientation labels and main dimensions
    ax, ay = L + 1.1, depth + 0.3
    fig.add_trace(go.Scatter3d(x=[ax, ax], y=[ay - 1.0, ay], z=[0.03, 0.03], mode="lines",
                               line=dict(color=COLORS["edge"], width=7), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Cone(x=[ax], y=[ay], z=[0.03], u=[0], v=[0.55], w=[0], anchor="tail",
                          sizemode="absolute", sizeref=0.55, colorscale=[[0, COLORS["edge"]], [1, COLORS["edge"]]],
                          showscale=False, hoverinfo="skip"))
    label_y_south = -1.45 if (case_number != 960 and src == 610) else -0.9
    fig.add_trace(go.Scatter3d(
        x=[ax, L / 2],
        y=[ay + 0.75, label_y_south - 0.3],
        z=[0.03, 0.03], mode="text", text=["<b>N</b>", "<b>South</b>"],
        textfont=dict(size=14, color=COLORS["edge"]), hoverinfo="skip", showlegend=False))

    if night_vent:  # schematic: the standard gives the fan capacity, not its position
        zc = z0 + H / 2
        for (x0_, y0_, x1_, y1_) in ((4.0, -1.45, 4.0, -0.12), (L + 0.12, W / 2, L + 1.45, W / 2)):  # in, out
            fig.add_trace(go.Scatter3d(
                x=[x0_, x1_], y=[y0_, y1_], z=[zc, zc], mode="lines+markers",
                line=dict(color=COLORS["airflow"], width=10),
                marker=dict(color=COLORS["airflow"], size=[1, 9], symbol="diamond"),
                hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   line=dict(color=COLORS["airflow"], width=8),
                                   name="Night vent fan 18:00-07:00 (schematic: air in, air out)",
                                   showlegend=True))

    hidden = dict(visible=False, showbackground=False)
    fig.update_layout(
        title="Test cell geometry" + (" (free-floating: no heating or cooling)" if free_float else ""),
        scene=dict(
            # Real proportions, scaled up so the building fills the frame.
            aspectmode="manual",
            aspectratio=dict(x=(L + 2 * pad) * SCALE, y=(depth + 2 * pad + 0.4) * SCALE, z=(z_top + 0.3) * SCALE),
            xaxis=hidden, yaxis=hidden, zaxis=hidden,
            camera=dict(projection=dict(type="orthographic"),
                        eye=(dict(x=0.95, y=-0.45, z=0.52) if src in (620, 630) and case_number != 960
                             else dict(x=0.55, y=-0.9, z=0.52)),
                        center=dict(x=0.0, y=0.0, z=-0.1)),
        ),
        legend=dict(orientation="h", y=0.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def geometry_caption(case_number: int, free_float: bool = False) -> str:
    src = 600 if case_number == 960 else _source_case(case_number)
    text = _geometry_caption(case_number, src)
    extra = {
        640: "Heating setback to 10 degC from 23:00 to 07:00 (no geometry change).",
        940: "Heating setback to 10 degC from 23:00 to 07:00 (no geometry change).",
        650: "Night ventilation 18:00-07:00 with the 1700 m3/h vent fan, no heating; the fan is drawn schematically.",
        660: "Low-e argon-filled glazing (Table 7-17).",
        670: "Single-pane glazing (Table 7-21).",
        680: "Wall and roof insulation increased to 0.25 m (wall) and 0.40 m (roof), drawn to scale (Table 7-25).",
        685: "20/20 degC thermostat (no geometry change).",
        695: "Wall and roof insulation increased as Case 680, with the 20/20 degC thermostat.",
    }.get(src)
    if case_number in (980, 995):
        extra = ("Wall and roof insulation increased as in Table 7-31, drawn to scale"
                 + (", with the 20/20 degC thermostat." if case_number == 995 else "."))
    if case_number == 985:
        extra = "20/20 degC thermostat (no geometry change)."
    if case_number == 960:
        extra = None
    if free_float:
        extra = ((extra + " ") if extra else "") + "Free-floating: no mechanical heating or cooling."
    return text + (" " + extra if extra else "")


def _geometry_caption(case_number: int, src: int) -> str:
    if case_number == 960:
        return ("Two zones (Figure 7-9): free-floating high-mass sun zone (2 m deep) south of the "
                "conditioned low-mass back zone, separated by the common wall; two 3 m x 2 m windows "
                "raised 0.3 m. Raised floor drawn above the ground (Section 7.2.1.5).")
    parts = ["8 m x 6 m x 2.7 m single zone"]
    if src in (620, 630):
        parts.append("one 3 m x 2 m window centred on the east wall and one on the west wall (Figure 7-4)")
    else:
        parts.append("two 3 m x 2 m south windows, sill 0.2 m (Figure 7-1)")
    if src == 610:
        parts.append("1.0 m overhang at roof level across the south wall (Figure 7-3)")
    if src == 630:
        parts.append("1.0 m overhang and 1.0 m side fins on each window (Figures 7-5, 7-6)")
    parts.append("raised floor drawn above the ground (Section 7.2.1.5)")
    return "; ".join(parts) + "."
