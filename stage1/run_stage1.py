#!/usr/bin/env python3
"""Stage 1: CadQuery .py → STL → Blender render (video.mp4, depth EXR, cameras.json)."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def setup_logging(log_path: Path, object_id: str) -> logging.Logger:
    logger = logging.getLogger(f"stage1.{object_id.replace('/', '_')}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def is_complete(output_dir: Path, num_frames: int, no_exr: bool = False) -> bool:
    if not (output_dir / "video.mp4").exists():
        return False
    if not (output_dir / "cameras.json").exists():
        return False
    if no_exr:
        return True
    depth_dir = output_dir / "depth"
    mask_dir  = output_dir / "mask"
    if not depth_dir.exists() or not mask_dir.exists():
        return False
    if len(list(depth_dir.glob("frame_*.exr"))) < num_frames:
        return False
    return len(list(mask_dir.glob("frame_*.exr"))) >= num_frames


def export_stl(script_path: Path, stl_path: Path, logger: logging.Logger) -> bool:
    namespace: dict = {"__file__": str(script_path)}
    try:
        code = script_path.read_text()
        exec(compile(code, str(script_path), "exec"), namespace)  # nosec B102
    except Exception as exc:
        logger.error("exec failed for %s: %s", script_path, exc)
        return False

    r = namespace.get("result") or namespace.get("r")
    if r is None:
        logger.error("No variable 'result' (or 'r') after executing %s", script_path)
        return False

    try:
        import cadquery as cq
        cq.exporters.export(r, str(stl_path))
    except Exception as exc:
        logger.error("STL export failed: %s", exc)
        return False

    if not stl_path.exists() or stl_path.stat().st_size == 0:
        logger.error("STL file empty or missing after export")
        return False

    logger.info("STL exported: %s (%d bytes)", stl_path, stl_path.stat().st_size)
    return True


def run_blender(
    stl_path: Path,
    aruco_marker: Path,
    output_dir: Path,
    render_script: Path,
    num_frames: int,
    fps: int,
    resolution: list,
    focal_length_mm: float,
    orbit_radius_factor: float,
    elevation_deg: float,
    aruco_size_mm: float,
    enable_blobs: bool,
    no_exr: bool,
    blender_exe: str,
    engine: str,
    logger: logging.Logger,
) -> bool:
    cmd = [
        blender_exe,
        "--background",
        "--python", str(render_script),
        "--",
        "--stl_path", str(stl_path),
        "--aruco_marker", str(aruco_marker),
        "--output_dir", str(output_dir),
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--resolution", str(resolution[0]), str(resolution[1]),
        "--focal_length_mm", str(focal_length_mm),
        "--orbit_radius_factor", str(orbit_radius_factor),
        "--elevation_deg", str(elevation_deg),
        "--aruco_size_mm", str(aruco_size_mm),
        "--engine", engine,
    ]
    if enable_blobs:
        cmd.append("--enable_blobs")
    if no_exr:
        cmd.append("--no_exr")
    logger.info("Blender cmd: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        logger.error("Blender timed out after 3600 s")
        return False
    except FileNotFoundError:
        logger.error("Blender executable not found: %s", blender_exe)
        return False
    except Exception as exc:
        logger.error("Blender subprocess error: %s", exc)
        return False

    if result.returncode != 0:
        logger.error("Blender exited %d", result.returncode)
        logger.error("STDOUT (tail): %s", result.stdout[-3000:])
        logger.error("STDERR (tail): %s", result.stderr[-3000:])
        return False

    logger.debug("Blender stdout: %s", result.stdout[-1000:])
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 1: CadQuery → STL → Blender render")
    parser.add_argument("--object_id", required=True,
                        help="Object ID relative to input_dir, e.g. 0000/00000007")
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="Root directory containing CadQuery .py scripts")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Root output directory (per-object subdir created automatically)")
    parser.add_argument("--aruco_marker", required=True, type=Path,
                        help="Path to ArUco marker PNG (200×200 px, DICT_4X4_50 ID 0)")
    parser.add_argument("--blender", default="blender",
                        help="Blender executable (default: blender from PATH)")
    parser.add_argument("--engine", default="BLENDER_EEVEE",
                        choices=["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"],
                        help="Render engine (use BLENDER_EEVEE_NEXT for Blender 4.x)")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480],
                        metavar=("W", "H"))
    parser.add_argument("--focal_length_mm", type=float, default=50.0)
    parser.add_argument("--orbit_radius_factor", type=float, default=3.5)
    parser.add_argument("--elevation_deg", type=float, default=30.0)
    parser.add_argument("--aruco_size_mm", type=float, default=50.0,
                        help="Physical side-length of the printed ArUco marker (mm)")
    parser.add_argument("--enable_blobs", action="store_true",
                        help="Scatter random background blobs around the object (off by default)")
    parser.add_argument("--no_exr", action="store_true",
                        help="Skip per-frame depth/mask EXRs (VLM-only renders, ~10x smaller)")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if all outputs already exist")
    args = parser.parse_args()

    render_script = Path(__file__).parent / "render_scene.py"
    if not render_script.exists():
        print(f"ERROR: render_scene.py not found at {render_script}", file=sys.stderr)
        sys.exit(1)

    script_path = args.input_dir / f"{args.object_id}.py"
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    if not args.aruco_marker.exists():
        print(f"ERROR: ArUco marker not found: {args.aruco_marker}", file=sys.stderr)
        sys.exit(1)

    obj_out = args.output_dir / args.object_id
    obj_out.mkdir(parents=True, exist_ok=True)
    if not args.no_exr:
        (obj_out / "depth").mkdir(exist_ok=True)

    logger = setup_logging(obj_out / "stage1.log", args.object_id)
    logger.info("Starting Stage 1 for object %s", args.object_id)

    if not args.force and is_complete(obj_out, args.num_frames, args.no_exr):
        logger.info("All outputs present — skipping (use --force to re-render)")
        return

    stl_path = obj_out / "object.stl"
    if args.force or not stl_path.exists():
        if not export_stl(script_path, stl_path, logger):
            logger.error("Aborting after STL export failure")
            sys.exit(2)
    else:
        logger.info("Reusing existing STL: %s", stl_path)

    ok = run_blender(
        stl_path=stl_path,
        aruco_marker=args.aruco_marker,
        output_dir=obj_out,
        render_script=render_script,
        num_frames=args.num_frames,
        fps=args.fps,
        resolution=args.resolution,
        focal_length_mm=args.focal_length_mm,
        orbit_radius_factor=args.orbit_radius_factor,
        elevation_deg=args.elevation_deg,
        aruco_size_mm=args.aruco_size_mm,
        enable_blobs=args.enable_blobs,
        no_exr=args.no_exr,
        blender_exe=args.blender,
        engine=args.engine,
        logger=logger,
    )
    if not ok:
        logger.error("Blender render failed")
        sys.exit(3)

    if not is_complete(obj_out, args.num_frames, args.no_exr):
        logger.error("Blender finished but expected outputs are missing")
        sys.exit(4)

    logger.info("Stage 1 complete for %s → %s", args.object_id, obj_out)


if __name__ == "__main__":
    main()
