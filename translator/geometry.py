"""Geometry helpers: curve chaining, arc midpoint, point math."""
from __future__ import annotations
import math
from typing import Optional


# Default endpoint-matching tolerance.
# DeepCAD coords are stored as 8-digit decimals (~1e-8 precision) but rounded
# from arbitrary Fusion floats — adjacent curves can drift by 1e-7..1e-6 cm.
# 1e-5 cm = 100 nm; well below any meaningful CAD feature.
TOL = 1e-5

# Loops smaller than this perimeter are considered degenerate and skipped.
MIN_LOOP_PERIMETER = 1e-7


def pt2d(p: dict) -> tuple[float, float]:
    """Extract (x, y) from a JSON point dict — z is ignored (sketch-local 2D)."""
    return float(p["x"]), float(p["y"])


def dist2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def curve_endpoints(curve: dict) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return (start, end) in 2D for a curve, or None if it's a closed loop (Circle3D)."""
    t = curve.get("type")
    if t == "Circle3D":
        return None
    if "start_point" in curve and "end_point" in curve:
        return pt2d(curve["start_point"]), pt2d(curve["end_point"])
    return None


def curve_length(curve: dict) -> float:
    """Approximate length of a curve segment (chord length is fine for chaining checks)."""
    e = curve_endpoints(curve)
    if e is None:
        # Circle3D: full circumference
        r = float(curve.get("radius", 0.0))
        return 2 * math.pi * r
    return dist2d(e[0], e[1])


def chain_curves(
    curves: list[dict],
    tol: float = TOL,
    relaxed: bool = False,
) -> Optional[list[tuple[dict, bool]]]:
    """Order curves into a continuous closed loop. Returns list of (curve, reversed_flag).

    Returns None if curves cannot be chained into a closed loop. Drops
    zero-length curves (start ≈ end) since they confuse chaining.

    If `relaxed=True`, also accepts a chain whose endpoints don't match exactly
    (within `10*tol`) — caller is expected to bridge the gap with a straight line.
    """
    if not curves:
        return []

    # Drop zero-length open curves; they cause spurious chaining failures.
    cleaned = []
    for c in curves:
        if c.get("type") == "Circle3D":
            r = float(c.get("radius", 0.0))
            if r > tol:
                cleaned.append(c)
            continue
        e = curve_endpoints(c)
        if e is None:
            continue
        if dist2d(e[0], e[1]) > tol * 0.5:
            cleaned.append(c)
    curves = cleaned
    if not curves:
        return None

    # Circle3D is its own loop. If a profile contains a circle, it should be the only curve.
    circles = [c for c in curves if c.get("type") == "Circle3D"]
    if circles:
        if len(curves) == 1:
            return [(curves[0], False)]
        # Mixed circle + other curves in same loop is unusual; bail.
        return None

    remaining = list(curves)
    close_tol = tol * (10.0 if relaxed else 1.0)

    def try_chain(start_curve_idx: int, start_reversed: bool) -> Optional[list[tuple[dict, bool]]]:
        rem = list(remaining)
        first = rem.pop(start_curve_idx)
        ends = curve_endpoints(first)
        if ends is None:
            return None
        sp, ep = ends
        if start_reversed:
            sp, ep = ep, sp
        loop_start = sp
        last = ep
        chain: list[tuple[dict, bool]] = [(first, start_reversed)]
        while rem:
            best_idx: Optional[int] = None
            best_rev = False
            best_d = float("inf")
            for i, c in enumerate(rem):
                ce = curve_endpoints(c)
                if ce is None:
                    continue
                cs, csend = ce
                d_fwd = dist2d(cs, last)
                d_rev = dist2d(csend, last)
                if d_fwd < d_rev and d_fwd < best_d:
                    best_idx, best_rev, best_d = i, False, d_fwd
                elif d_rev <= d_fwd and d_rev < best_d:
                    best_idx, best_rev, best_d = i, True, d_rev
            if best_idx is None or best_d >= tol:
                return None
            c = rem.pop(best_idx)
            ce = curve_endpoints(c)
            if best_rev:
                chain.append((c, True))
                last = ce[0]
            else:
                chain.append((c, False))
                last = ce[1]
        if dist2d(last, loop_start) > close_tol:
            return None
        return chain

    # Try each curve as starting curve, both orientations.
    for i in range(len(remaining)):
        for rev in (False, True):
            res = try_chain(i, rev)
            if res is not None:
                return res
    return None


def arc_midpoint(curve: dict, reversed_flag: bool) -> Optional[tuple[float, float]]:
    """Compute the geometric midpoint of an Arc3D in sketch-local 2D.

    Tries angle-based first (uses reference_vector + sweep), then chord-perpendicular.
    Returns None only if both strategies degenerate (collinear start/end/center).
    """
    cx, cy = pt2d(curve["center_point"])
    radius = float(curve["radius"])
    start_angle = float(curve.get("start_angle", 0.0))
    end_angle = float(curve.get("end_angle", 0.0))
    sweep = end_angle - start_angle
    # Normalize sweep into (0, 2π].
    if sweep <= 0:
        sweep += 2 * math.pi
    if sweep <= 0 or radius <= 0:
        return None

    angle_based = _arc_midpoint_from_angles(curve, sweep)
    if angle_based is not None and _on_circle(angle_based, (cx, cy), radius):
        return angle_based

    chord_based = _arc_midpoint_from_chord(curve, sweep)
    if chord_based is not None:
        return chord_based
    return angle_based  # last resort, even if slightly off-circle


def _on_circle(pt: tuple[float, float], center: tuple[float, float], radius: float) -> bool:
    return abs(dist2d(pt, center) - radius) < max(1e-6, radius * 1e-4)


def _arc_midpoint_from_angles(curve: dict, sweep: float) -> Optional[tuple[float, float]]:
    cx, cy = pt2d(curve["center_point"])
    radius = float(curve["radius"])
    start_angle = float(curve.get("start_angle", 0.0))
    ref = curve.get("reference_vector")
    if ref is None:
        return None
    rvx, rvy = float(ref["x"]), float(ref["y"])
    rlen = math.hypot(rvx, rvy)
    if rlen < 1e-12:
        return None
    rvx, rvy = rvx / rlen, rvy / rlen
    pvx, pvy = -rvy, rvx  # CCW perpendicular in 2D
    mid_t = start_angle + sweep / 2.0
    cos_t, sin_t = math.cos(mid_t), math.sin(mid_t)
    mx = cx + radius * (cos_t * rvx + sin_t * pvx)
    my = cy + radius * (cos_t * rvy + sin_t * pvy)
    return mx, my


def _arc_midpoint_from_chord(curve: dict, sweep: float) -> Optional[tuple[float, float]]:
    sp = pt2d(curve["start_point"])
    ep = pt2d(curve["end_point"])
    cx, cy = pt2d(curve["center_point"])
    radius = float(curve["radius"])
    # chord midpoint
    mx = (sp[0] + ep[0]) / 2.0
    my = (sp[1] + ep[1]) / 2.0
    dx, dy = mx - cx, my - cy
    dlen = math.hypot(dx, dy)
    if dlen < 1e-12:
        # Diameter chord — use perpendicular to chord for direction.
        cdx, cdy = ep[0] - sp[0], ep[1] - sp[1]
        clen = math.hypot(cdx, cdy)
        if clen < 1e-12:
            return None
        ux, uy = -cdy / clen, cdx / clen  # 90° CCW
        return cx + ux * radius, cy + uy * radius
    ux, uy = dx / dlen, dy / dlen
    if sweep <= math.pi:
        return cx + ux * radius, cy + uy * radius
    return cx - ux * radius, cy - uy * radius


def loop_perimeter(chain: list[tuple[dict, bool]]) -> float:
    """Sum of curve lengths in a chained loop."""
    return sum(curve_length(c) for c, _ in chain)
