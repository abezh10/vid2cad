# Stage 1 — CadQuery → Blender Orbit Render

Converts CadQuery `.py` scripts into a smooth orbit video, per-frame depth maps,
and camera poses.  Outputs feed directly into Stage 2 (metric point cloud via
TSDF fusion).

## Overview

```
CadQuery .py
    └─ run_stage1.py ──→ object.stl   (via cadquery)
                    └──→ Blender subprocess (render_scene.py)
                              ├─ video.mp4          (60 frames, 24 fps, 640×480)
                              ├─ depth/frame_NNNN.exr  (32-bit float, mm)
                              ├─ cameras.json        (per-frame intrinsics + extrinsics)
                              └─ metadata.json       (bounding box, resolution)
```

## Directory layout

```
stage1/
  run_stage1.py          main orchestration script
  render_scene.py        Blender Python script (never import directly)
  validate_stage1.py     post-run output checker
  submit_stage1.sh       SLURM array job submission helper
  setup_stage1_env.sh    conda environment creation
  generate_aruco.py      one-off ArUco marker PNG generator
  README.md              this file
```

---

## 1. Environment setup

Run the setup script once.  It handles everything:

```bash
bash stage1/setup_stage1_env.sh
```

What it does:

1. **Python packages** — uses `module load python/3.13.5`, which already has
   `cadquery 2.7.0` and `numpy`.  Installs `opencv-python` to your user
   site-packages (`~/.local/lib/python3.13/`) if not already present.
   No conda needed; conda is not available on FarmShare.

2. **Blender** — downloads the Blender 3.6 LTS portable binary to
   `/scratch/users/abzhong/blender/` (self-contained, no `module load`
   required).  Skip with `--skip-blender` if you already have a copy.

```bash
# Custom Blender install path:
bash stage1/setup_stage1_env.sh --blender-dir /scratch/users/abzhong/blender

# If Blender is already downloaded:
bash stage1/setup_stage1_env.sh --skip-blender
```

OpenEXR depth output is handled natively by Blender's compositor — no
extra Python OpenEXR package is needed.

**Blender version notes:**
- Blender 3.x → `--engine BLENDER_EEVEE` (default)
- Blender 4.x → `--engine BLENDER_EEVEE_NEXT`

---

## 2. Generate ArUco marker

This is a one-time step.  The marker PNG is shared across all objects.

```bash
conda activate vid2cad-stage1
python3 stage1/generate_aruco.py --output aruco_marker.png
```

Produces a 200×200 px PNG representing a physical 50 mm × 50 mm square
(DICT_4X4_50, ID 0).  The Blender scene places the marker 80 mm to the right
of each object's bounding box.

---

## 3. Smoke test (interactive, no sbatch)

Test a single object before submitting the full array:

```bash
module load python/3.13.5

python3 stage1/run_stage1.py \
    --object_id  0000/00000007 \
    --input_dir  deepcad-data/cadquery_py \
    --output_dir /scratch/users/abzhong/stage1_output \
    --aruco_marker aruco_marker.png \
    --blender /scratch/users/abzhong/blender/blender
```

Inspect outputs:
```
stage1_output/0000/00000007/
  video.mp4
  cameras.json
  metadata.json
  object.stl
  stage1.log
  depth/
    frame_0001.exr … frame_0060.exr
```

---

## 4. SLURM array submission

```bash
bash stage1/submit_stage1.sh \
    --input_dir    deepcad-data/cadquery_py \
    --output_dir   /scratch/users/abzhong/stage1_output \
    --aruco_marker aruco_marker.png \
    --conda_env    vid2cad-stage1 \
    --blender      blender \
    --engine       BLENDER_EEVEE \
    --time         01:00:00
```

The script:
1. Finds all `.py` scripts under `--input_dir`.
2. Writes `stage1_output/stage1_input_manifest.txt` (one path per line).
3. Writes `stage1_output/stage1_array.sbatch` (fully resolved, safe to inspect).
4. Submits the array and prints the job ID.

Per-task logs land in `logs/stage1_<ArrayJobID>_<TaskID>.{out,err}`.
Per-object logs land in `<output_dir>/<object_id>/stage1.log`.

**Re-runs are safe:** objects whose `video.mp4`, `cameras.json`, and all 60
depth EXRs are already present are skipped automatically.  Use `--force` in
`run_stage1.py` to override.

### Monitoring

```bash
squeue -u $USER
sacct -j <JobID> --format=JobID,State,Elapsed,MaxRSS
```

---

## 5. Validation

```bash
python3 stage1/validate_stage1.py \
    --output_dir /scratch/users/abzhong/stage1_output \
    --verbose
```

Prints success/failure counts, failure breakdown, and object bounding-box
statistics.  Saves `stage1_output/stage1_summary.json`.

---

## 6. Output format reference

### `cameras.json`

Array of 60 entries, one per frame:

```json
{
  "frame": 1,
  "position_mm": [x, y, z],
  "world_to_camera": [[...4×4 float matrix...]],
  "convention": "opencv",
  "fx": 888.9,
  "fy": 888.9,
  "cx": 320.0,
  "cy": 240.0,
  "width": 640,
  "height": 480
}
```

**Convention:** `world_to_camera` uses the **OpenCV** convention (camera looks
along +Z, Y-axis points down).  To get the camera-to-world matrix, invert it.

Focal length in pixels is computed as `f_mm / sensor_width_mm * image_width_px`
with a 36 mm full-frame sensor.

### `depth/frame_NNNN.exr`

Single-channel 32-bit float EXR.  Pixel value = distance from the camera
plane to the surface **along the camera Z axis**, in **millimetres**.
Background pixels (no geometry) carry Blender's far-clip value (~10⁷ mm) and
should be masked out downstream.

### `metadata.json`

```json
{
  "num_frames": 60,
  "resolution": [640, 480],
  "bbox_min_mm": [x, y, z],
  "bbox_max_mm": [x, y, z],
  "bbox_dims_mm": [dx, dy, dz],
  "longest_dim_mm": d,
  "orbit_radius_mm": r
}
```

---

## 7. Feeding Stage 2

Stage 2 (TSDF fusion → metric point cloud) needs:

| Stage 1 output | Stage 2 input role |
|---|---|
| `depth/frame_NNNN.exr` | per-frame depth maps |
| `cameras.json` | camera intrinsics + extrinsics |
| `metadata.json` | object scale / voxel grid sizing |

Depth values are already in millimetres. `cameras.json` uses the OpenCV
convention expected by most TSDF libraries (Open3D, etc.).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `blender: command not found` | Run `setup_stage1_env.sh` to download the portable binary, then pass `--blender /scratch/users/abzhong/blender/blender` |
| `BLENDER_EEVEE` not recognized | Pass `--engine BLENDER_EEVEE_NEXT` (Blender 4.x) |
| STL export fails with cadquery error | Check that `module load python/3.13.5` and `conda activate vid2cad-stage1` were run first |
| Depth EXRs present but all-zero | Z pass needs `use_pass_z = True` on the view layer — already set in `render_scene.py` |
| `No variable 'result' (or 'r') after executing` | CadQuery script does not assign its result to `result` or `r` |
