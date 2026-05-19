"""Emit CADQuery Python source from parsed JSON entities."""
from __future__ import annotations
import math
from typing import Optional

from .geometry import (
    chain_curves,
    arc_midpoint,
    pt2d,
    dist2d,
    loop_perimeter,
    MIN_LOOP_PERIMETER,
)


SUPPORTED_CURVE_TYPES = {"Line3D", "Arc3D", "Circle3D"}

# Extrudes whose total span is below this are considered no-ops (would create
# zero-volume bodies that fail downstream booleans).
MIN_EXTRUDE_DISTANCE = 1e-7


class UnsupportedFeatureError(Exception):
    pass


def fmt(v: float) -> str:
    """Format a float for inclusion in generated code."""
    if v == 0:
        return "0.0"
    return repr(float(v))


def emit_workplane_def(name: str, sketch: dict, normal_offset: float = 0.0) -> list[str]:
    """Emit code defining `name` as a cq.Workplane from a sketch's transform.

    `normal_offset` shifts the plane origin along its normal (z_axis) by that amount.
    Used by two-sided extrudes to start the workplane on the "bottom" face.
    """
    t = sketch.get("transform") or {}
    origin = t.get("origin") or {"x": 0, "y": 0, "z": 0}
    x_axis = t.get("x_axis") or {"x": 1, "y": 0, "z": 0}
    z_axis = t.get("z_axis") or {"x": 0, "y": 0, "z": 1}
    ox, oy, oz = float(origin["x"]), float(origin["y"]), float(origin["z"])
    xx, xy, xz = float(x_axis["x"]), float(x_axis["y"]), float(x_axis["z"])
    zx, zy, zz = float(z_axis["x"]), float(z_axis["y"]), float(z_axis["z"])
    if normal_offset:
        ox += zx * normal_offset
        oy += zy * normal_offset
        oz += zz * normal_offset
    plane_var = f"plane_{name}"
    return [
        f"{plane_var} = cq.Plane(",
        f"    origin=cq.Vector({fmt(ox)}, {fmt(oy)}, {fmt(oz)}),",
        f"    xDir=cq.Vector({fmt(xx)}, {fmt(xy)}, {fmt(xz)}),",
        f"    normal=cq.Vector({fmt(zx)}, {fmt(zy)}, {fmt(zz)}),",
        f")",
        f"{name} = cq.Workplane({plane_var})",
    ]


def emit_loop(wp_var: str, curves: list[dict]) -> Optional[list[str]]:
    """Emit chained sketch operations for a single loop's curves.

    Returns lines that mutate `wp_var` in place, or None if the loop is degenerate
    or unsupported (caller decides whether to skip or fail). Raises
    UnsupportedFeatureError only for curves we explicitly do not handle.
    """
    if not curves:
        return None
    for c in curves:
        if c.get("type") not in SUPPORTED_CURVE_TYPES:
            raise UnsupportedFeatureError(f"curve type {c.get('type')!r}")

    chained = chain_curves(curves)
    bridge_to_close = False
    if chained is None:
        # Try a relaxed pass; if it works, bridge any small gap with a straight
        # line so CADQuery still gets a closed wire.
        chained = chain_curves(curves, relaxed=True)
        if chained is None:
            return None
        bridge_to_close = True

    if loop_perimeter(chained) < MIN_LOOP_PERIMETER:
        return None

    lines: list[str] = []
    # Circle3D as standalone loop.
    if len(chained) == 1 and chained[0][0].get("type") == "Circle3D":
        c = chained[0][0]
        cx, cy = pt2d(c["center_point"])
        r = float(c["radius"])
        if r <= 0:
            return None
        lines.append(f"{wp_var} = {wp_var}.moveTo({fmt(cx)}, {fmt(cy)}).circle({fmt(r)})")
        return lines

    # Line/Arc chain.
    first_curve, first_rev = chained[0]
    sp, ep = pt2d(first_curve["start_point"]), pt2d(first_curve["end_point"])
    if first_rev:
        sp, ep = ep, sp
    start_x, start_y = sp
    lines.append(f"{wp_var} = {wp_var}.moveTo({fmt(start_x)}, {fmt(start_y)})")
    lines.extend(_emit_curve_segment(wp_var, first_curve, first_rev))
    for curve, rev in chained[1:]:
        lines.extend(_emit_curve_segment(wp_var, curve, rev))

    # If we used the relaxed chain, the final point may not exactly equal the
    # start. Bridge with an explicit lineTo before closing.
    if bridge_to_close:
        last_curve, last_rev = chained[-1]
        last_end = pt2d(last_curve["start_point"] if last_rev else last_curve["end_point"])
        if dist2d(last_end, sp) > 0:
            lines.append(f"{wp_var} = {wp_var}.lineTo({fmt(start_x)}, {fmt(start_y)})")
    lines.append(f"{wp_var} = {wp_var}.close()")
    return lines


def _emit_curve_segment(wp_var: str, curve: dict, reversed_flag: bool) -> list[str]:
    t = curve["type"]
    if t == "Line3D":
        end = pt2d(curve["start_point"] if reversed_flag else curve["end_point"])
        return [f"{wp_var} = {wp_var}.lineTo({fmt(end[0])}, {fmt(end[1])})"]
    if t == "Arc3D":
        mid = arc_midpoint(curve, reversed_flag)
        end = pt2d(curve["start_point"] if reversed_flag else curve["end_point"])
        if mid is None:
            # Degenerate arc — emit a straight line to keep the wire closed.
            return [f"{wp_var} = {wp_var}.lineTo({fmt(end[0])}, {fmt(end[1])})"]
        return [
            f"{wp_var} = {wp_var}.threePointArc("
            f"({fmt(mid[0])}, {fmt(mid[1])}), "
            f"({fmt(end[0])}, {fmt(end[1])}))"
        ]
    raise UnsupportedFeatureError(f"curve type {t!r}")


def emit_profile(wp_var: str, sketch: dict, profile_id: str) -> tuple[list[str], int]:
    """Emit all loops for a profile onto the workplane variable.

    Returns (lines, n_loops_emitted). Skips degenerate/unchainable loops silently
    so a single bad loop does not kill the whole profile. Caller should treat
    n_loops_emitted == 0 as a profile-level failure.
    """
    profile = sketch["profiles"].get(profile_id)
    if profile is None:
        raise UnsupportedFeatureError(f"profile {profile_id!r} not found in sketch")
    lines: list[str] = []
    n_emitted = 0
    for loop in profile["loops"]:
        curves = loop.get("profile_curves") or []
        try:
            loop_lines = emit_loop(wp_var, curves)
        except UnsupportedFeatureError:
            # Unsupported curve in this loop — skip it; if the outer loop
            # required this, the result will be wrong but at least the rest
            # of the design has a chance.
            continue
        if not loop_lines:
            continue
        lines.extend(loop_lines)
        n_emitted += 1
    return lines, n_emitted


def emit_extrude_feature(
    extrude: dict,
    sketches: dict[str, dict],
    result_var: str = "result",
    sketch_var_counter: list[int] = None,
) -> list[str]:
    """Emit code for an ExtrudeFeature.

    `sketches` maps sketch UUID → sketch entity dict.
    `result_var` is the name of the running solid variable. None means uninitialized.
    `sketch_var_counter` is a single-element list used to generate unique names.
    """
    if sketch_var_counter is None:
        sketch_var_counter = [0]

    profiles = extrude.get("profiles") or []
    if not profiles:
        raise UnsupportedFeatureError("extrude has no profiles")

    operation = extrude.get("operation", "NewBodyFeatureOperation")
    extent_type = extrude.get("extent_type", "OneSideFeatureExtentType")

    # Distances + tapers
    e1 = extrude.get("extent_one") or {}
    e2 = extrude.get("extent_two") or {}
    d1 = float((e1.get("distance") or {}).get("value", 0.0))
    d2 = float((e2.get("distance") or {}).get("value", 0.0))
    taper1 = float((e1.get("taper_angle") or {}).get("value", 0.0))

    by_sketch: dict[str, list[str]] = {}
    for p in profiles:
        by_sketch.setdefault(p["sketch"], []).append(p["profile"])

    # Compute workplane offset & total extrude distance based on extent_type.
    if extent_type == "OneSideFeatureExtentType":
        normal_offset = 0.0
        total_distance = d1
        both_sides = False
    elif extent_type == "SymmetricFeatureExtentType":
        normal_offset = 0.0
        total_distance = d1
        both_sides = True
    elif extent_type == "TwoSidesFeatureExtentType":
        normal_offset = -d2
        total_distance = d1 + d2
        both_sides = False
    else:
        raise UnsupportedFeatureError(f"extent_type {extent_type!r}")

    # Skip zero-thickness extrudes — they produce empty bodies and break later booleans.
    span = abs(total_distance) * (2.0 if both_sides else 1.0)
    if span < MIN_EXTRUDE_DISTANCE:
        raise UnsupportedFeatureError("zero-thickness extrude")

    lines: list[str] = []
    body_var: Optional[str] = None
    any_profile_emitted = False
    for sketch_uuid, profile_ids in by_sketch.items():
        sketch = sketches.get(sketch_uuid)
        if sketch is None:
            continue
        sketch_var_counter[0] += 1
        wp_name = f"wp_{sketch_var_counter[0]}"
        wp_lines: list[str] = list(emit_workplane_def(wp_name, sketch, normal_offset=normal_offset))
        wp_lines.insert(0, "")
        n_total = 0
        for pid in profile_ids:
            try:
                pl, n = emit_profile(wp_name, sketch, pid)
            except UnsupportedFeatureError:
                continue
            wp_lines.extend(pl)
            n_total += n
        if n_total == 0:
            sketch_var_counter[0] -= 1  # unused name; rewind so next gets the same
            continue

        extr_expr = _emit_extrude_call(wp_name, total_distance, both_sides, taper1)
        wp_lines.append(
            f"body_{sketch_var_counter[0]} = {extr_expr}"
            if body_var is None
            else f"body_{sketch_var_counter[0]} = {extr_expr}"
        )
        lines.extend(wp_lines)
        this_body = f"body_{sketch_var_counter[0]}"
        if body_var is None:
            body_var = this_body
        else:
            lines.append(f"{body_var} = {body_var}.union({this_body})")
        any_profile_emitted = True

    if not any_profile_emitted or body_var is None:
        raise UnsupportedFeatureError("no usable profiles in extrude")

    if operation in ("NewBodyFeatureOperation", "JoinFeatureOperation"):
        lines.append(
            f"{result_var} = {body_var} if {result_var} is None else {result_var}.union({body_var})"
        )
    elif operation == "CutFeatureOperation":
        lines.append(
            f"{result_var} = {result_var}.cut({body_var}) if {result_var} is not None else {body_var}"
        )
    elif operation == "IntersectFeatureOperation":
        lines.append(
            f"{result_var} = {result_var}.intersect({body_var}) if {result_var} is not None else {body_var}"
        )
    else:
        raise UnsupportedFeatureError(f"operation {operation!r}")

    return lines


def _emit_extrude_call(
    wp_var: str,
    total_distance: float,
    both_sides: bool,
    taper_rad: float,
) -> str:
    args = fmt(total_distance)
    if both_sides:
        args += ", both=True"
    if taper_rad:
        args += f", taper={fmt(math.degrees(taper_rad))}"
    return f"{wp_var}.extrude({args})"
