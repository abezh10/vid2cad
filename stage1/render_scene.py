"""Blender render script for Stage 1.

Invoked via:
    blender --background --python render_scene.py -- [args]

Never import this file directly outside of a Blender subprocess.
bpy is only available inside Blender's embedded Python interpreter.
"""

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path


# DeepCAD coordinates are unitless but read as metres in Blender (a value of
# 0.254 corresponds to a 10-inch / 254 mm dimension). All "_mm" metadata
# fields therefore store *scene units*, not literal millimetres.
SCENE_UNIT_MM = 1000.0   # 1 scene unit (Blender metre) == 1000 mm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--stl_path",            required=True, type=Path)
    p.add_argument("--aruco_marker",        required=True, type=Path)
    p.add_argument("--output_dir",          required=True, type=Path)
    p.add_argument("--num_frames",          type=int,   default=60)
    p.add_argument("--fps",                 type=int,   default=24)
    p.add_argument("--resolution",          nargs=2, type=int, default=[640, 480])
    p.add_argument("--focal_length_mm",     type=float, default=50.0)
    p.add_argument("--orbit_radius_factor", type=float, default=3.5)
    p.add_argument("--elevation_deg",       type=float, default=30.0,
                   help="Centre of the random elevation range (±20°)")
    p.add_argument("--aruco_size_mm",       type=float, default=50.0,
                   help="Physical side-length of the printed ArUco marker (mm)")
    p.add_argument("--enable_blobs",        action="store_true",
                   help="Scatter randomised background blobs around the object (off by default)")
    p.add_argument("--no_exr",              action="store_true",
                   help="Skip per-frame depth/mask EXR outputs (VLM-only renders; "
                        "~10x smaller per object). cameras.json is still written.")
    p.add_argument("--engine",              default="BLENDER_EEVEE")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Randomness helpers
# ---------------------------------------------------------------------------

def derive_seed(output_dir: Path) -> int:
    """Deterministic seed from the object's output path."""
    h = hashlib.md5(str(output_dir).encode()).digest()
    return int.from_bytes(h[:4], "little")


def kelvin_to_rgb(temp_k: float) -> tuple:
    """Tanner Helland approximation: colour temperature (K) → linear RGB."""
    t = max(1000.0, min(40000.0, temp_k)) / 100.0
    if t <= 66:
        r = 1.0
        g = max(0.0, min(1.0, 0.3900815787690965 * math.log(t) - 0.6318414437169469))
        b = 0.0 if t <= 19 else max(0.0, min(1.0,
            0.5432067891510679 * math.log(t - 10) - 1.1962541092242649))
    else:
        r = max(0.0, min(1.0, 1.2930649907372248 * ((t - 60) ** -0.1332047592)))
        g = max(0.0, min(1.0, 1.1298908609037744 * ((t - 60) ** -0.0755148492)))
        b = 1.0
    return (r, g, b)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def clear_scene():
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for col in (bpy.data.meshes, bpy.data.materials,
                bpy.data.cameras, bpy.data.lights, bpy.data.images):
        for block in list(col):
            col.remove(block)


def import_stl(stl_path: Path):
    import bpy
    try:
        bpy.ops.import_mesh.stl(filepath=str(stl_path))
    except AttributeError:
        bpy.ops.wm.stl_import(filepath=str(stl_path))
    obj = bpy.context.selected_objects[0]
    obj.name = "CADObject"
    return obj


def center_object(obj):
    """Translate the object so its geometry centroid sits at the world origin."""
    import bpy
    import mathutils
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center  = sum(corners, mathutils.Vector()) / 8.0
    obj.location -= center
    bpy.ops.object.transform_apply(location=True)


def world_bbox(obj):
    """Return (min_x, max_x, min_y, max_y, min_z, max_z) in world space."""
    import mathutils
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


# ---------------------------------------------------------------------------
# Scene element setup — all randomised
# ---------------------------------------------------------------------------

def setup_world_sky(rng: random.Random):
    """Replace the world shader with a randomised Nishita procedural sky."""
    import bpy
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type        = "NISHITA"
    sky.sun_elevation   = math.radians(rng.uniform(5.0, 80.0))
    sky.sun_rotation    = math.radians(rng.uniform(0.0, 360.0))
    sky.turbidity       = rng.uniform(2.0, 10.0)
    sky.ozone_density   = rng.uniform(0.0, 1.0)
    sky.air_density     = rng.uniform(0.5, 1.5)
    sky.dust_density    = rng.uniform(0.0, 1.0)
    sky.location        = (-300, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = rng.uniform(0.7, 1.5)
    bg.location = (0, 0)

    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (300, 0)

    links.new(sky.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_lighting(rng: random.Random):
    """Main sun + optional fill and rim sun lamps, all randomised.

    Using SUN lamps throughout avoids distance-falloff energy-scaling issues
    when objects span many orders of magnitude in scene units.
    """
    import bpy

    def add_sun(name, energy, temp_k, elev_rad, az_rad, angle_deg):
        bpy.ops.object.light_add(type="SUN")
        lamp = bpy.context.active_object
        lamp.name = name
        lamp.data.energy = energy
        lamp.data.color  = kelvin_to_rgb(temp_k)
        lamp.data.angle  = math.radians(angle_deg)
        lamp.rotation_euler = (math.pi / 2.0 - elev_rad, 0.0, az_rad)
        return lamp

    # --- Main sun ---
    sun_az   = math.radians(rng.uniform(0, 360))
    sun_elev = math.radians(rng.uniform(15, 75))
    add_sun(
        name      = "SunMain",
        energy    = rng.uniform(1.5, 4.5),
        temp_k    = rng.uniform(2700, 8000),
        elev_rad  = sun_elev,
        az_rad    = sun_az,
        angle_deg = rng.uniform(0.5, 3.0),
    )

    # --- Optional fill lamp (probability 0.7) ---
    if rng.random() < 0.7:
        fill_az   = sun_az + math.pi + math.radians(rng.uniform(-45, 45))
        fill_elev = math.radians(rng.uniform(10, 50))
        add_sun(
            name      = "FillLight",
            energy    = rng.uniform(0.1, 1.2),
            temp_k    = rng.uniform(4000, 9500),
            elev_rad  = fill_elev,
            az_rad    = fill_az,
            angle_deg = rng.uniform(5, 30),   # soft, diffuse fill
        )

    # --- Optional rim lamp (probability 0.5) ---
    if rng.random() < 0.5:
        rim_az   = sun_az + math.radians(rng.uniform(110, 250))
        rim_elev = math.radians(rng.uniform(30, 80))
        add_sun(
            name      = "RimLight",
            energy    = rng.uniform(0.5, 2.5),
            temp_k    = rng.choice([2700, 3200, 6500, 8000, 10000]),
            elev_rad  = rim_elev,
            az_rad    = rim_az,
            angle_deg = rng.uniform(1, 10),
        )


def add_aruco_plane(aruco_path: Path, bbox, aruco_size_mm: float, longest_dim: float,
                    ground_z: float, orbit_radius: float):
    """Emission-textured plane at a fixed physical size, placed flat on the ground.

    aruco_size_mm: fixed side-length matching the real printed marker.
    gap from right bbox edge = 0.2 × longest_dim (scales with object so marker stays visible).
    """
    import bpy
    min_x, max_x = bbox[0], bbox[1]
    marker_size = aruco_size_mm / SCENE_UNIT_MM   # mm → scene units (metres)
    gap         = 0.2 * longest_dim
    cx          = max_x + gap + marker_size * 0.5
    # Lift the marker 5 % of orbit_radius above the ground plane. A fixed or
    # longest_dim-proportional offset collapses below EEVEE's ~16-bit depth
    # precision at camera distance, causing z-fighting at all scene scales.
    # marker_z    = ground_z + 0.05 * orbit_radius
    marker_z    = ground_z + 0.01 * orbit_radius

    bpy.ops.mesh.primitive_plane_add(size=marker_size, location=(cx, 0.0, marker_z))
    plane = bpy.context.active_object
    plane.name = "ArUcoMarker"

    mat   = bpy.data.materials.new("ArUcoMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    tex_coord = nodes.new("ShaderNodeTexCoord");  tex_coord.location = (-600, 0)
    img_tex   = nodes.new("ShaderNodeTexImage");  img_tex.location   = (-300, 0)
    emission  = nodes.new("ShaderNodeEmission");  emission.location  = (0,    0)
    out       = nodes.new("ShaderNodeOutputMaterial"); out.location   = (300,  0)
    emission.inputs["Strength"].default_value = 1.0

    links.new(tex_coord.outputs["UV"],        img_tex.inputs["Vector"])
    links.new(img_tex.outputs["Color"],       emission.inputs["Color"])
    links.new(emission.outputs["Emission"],   out.inputs["Surface"])

    img_tex.image = bpy.data.images.load(str(aruco_path))
    # Closest-neighbour sampling keeps the ArUco squares crisp at any zoom
    # level. Default linear filtering blurs the bitmap into horizontal
    # moiré stripes that shift frame-to-frame as the camera orbits.
    img_tex.interpolation = "Closest"
    plane.data.materials.append(mat)
    return plane


def _ground_wood(nodes, links, bsdf, rng):
    """Wood grain: noise-distorted wave rings → color ramp."""
    tc = nodes.new("ShaderNodeTexCoord"); tc.location = (-800, 0)
    mp = nodes.new("ShaderNodeMapping");  mp.location = (-600, 0)
    nz = nodes.new("ShaderNodeTexNoise"); nz.location = (-400, 150)
    wv = nodes.new("ShaderNodeTexWave");  wv.location = (-200, 0)
    cr = nodes.new("ShaderNodeValToRGB"); cr.location = (   0, 0)

    links.new(tc.outputs["Object"],  mp.inputs["Vector"])
    links.new(mp.outputs["Vector"],  nz.inputs["Vector"])
    links.new(mp.outputs["Vector"],  wv.inputs["Vector"])
    links.new(nz.outputs["Fac"],     wv.inputs["Distortion"])
    links.new(wv.outputs["Fac"],     cr.inputs["Fac"])
    links.new(cr.outputs["Color"],   bsdf.inputs["Base Color"])

    mp.inputs["Scale"].default_value      = (rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), 1.0)
    wv.wave_type                          = "RINGS"
    wv.inputs["Scale"].default_value      = rng.uniform(4,  10)
    wv.inputs["Detail"].default_value     = rng.uniform(2,   6)
    nz.inputs["Scale"].default_value      = rng.uniform(5,  15)
    nz.inputs["Detail"].default_value     = rng.uniform(4,   8)
    nz.inputs["Roughness"].default_value  = rng.uniform(0.5, 0.9)

    palettes = [
        ((0.38, 0.19, 0.06), (0.65, 0.38, 0.14)),  # oak
        ((0.15, 0.07, 0.02), (0.40, 0.20, 0.07)),  # dark walnut
        ((0.55, 0.38, 0.18), (0.80, 0.62, 0.38)),  # pine/birch
        ((0.60, 0.40, 0.20), (0.82, 0.62, 0.40)),  # maple
    ]
    dark, light = rng.choice(palettes)
    cr.color_ramp.elements[0].color        = (*dark,  1.0)
    cr.color_ramp.elements[1].color        = (*light, 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.15, 0.45)


def _ground_marble(nodes, links, bsdf, rng):
    """Marble: noise-distorted wave bands → color ramp with vein contrast."""
    tc = nodes.new("ShaderNodeTexCoord"); tc.location = (-800, 0)
    mp = nodes.new("ShaderNodeMapping");  mp.location = (-600, 0)
    nz = nodes.new("ShaderNodeTexNoise"); nz.location = (-400, 150)
    wv = nodes.new("ShaderNodeTexWave");  wv.location = (-200, 0)
    cr = nodes.new("ShaderNodeValToRGB"); cr.location = (   0, 0)

    links.new(tc.outputs["Object"],  mp.inputs["Vector"])
    links.new(mp.outputs["Vector"],  nz.inputs["Vector"])
    links.new(mp.outputs["Vector"],  wv.inputs["Vector"])
    links.new(nz.outputs["Fac"],     wv.inputs["Distortion"])
    links.new(wv.outputs["Fac"],     cr.inputs["Fac"])
    links.new(cr.outputs["Color"],   bsdf.inputs["Base Color"])

    mp.inputs["Scale"].default_value      = (rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), 1.0)
    wv.wave_type                          = "BANDS"
    wv.inputs["Scale"].default_value      = rng.uniform(3,   8)
    wv.inputs["Detail"].default_value     = rng.uniform(4,  10)
    nz.inputs["Scale"].default_value      = rng.uniform(3,   8)
    nz.inputs["Detail"].default_value     = rng.uniform(6,  12)
    nz.inputs["Roughness"].default_value  = rng.uniform(0.5, 0.8)

    palettes = [
        ((0.85, 0.83, 0.80), (0.20, 0.18, 0.16)),  # white/grey marble
        ((0.12, 0.11, 0.10), (0.80, 0.78, 0.75)),  # black marble
        ((0.78, 0.70, 0.55), (0.92, 0.88, 0.75)),  # travertine
        ((0.62, 0.72, 0.65), (0.92, 0.90, 0.85)),  # green marble
    ]
    base, vein = rng.choice(palettes)
    cr.color_ramp.elements[0].color        = (*base, 1.0)
    cr.color_ramp.elements[1].color        = (*vein, 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.05, 0.20)


def _ground_metal(nodes, links, bsdf, rng):
    """Brushed metal: wave bands scaled along one axis, fully metallic BSDF."""
    tc = nodes.new("ShaderNodeTexCoord"); tc.location = (-500, 0)
    mp = nodes.new("ShaderNodeMapping");  mp.location = (-300, 0)
    wv = nodes.new("ShaderNodeTexWave");  wv.location = (-100, 0)
    cr = nodes.new("ShaderNodeValToRGB"); cr.location = ( 100, 0)

    links.new(tc.outputs["Object"],  mp.inputs["Vector"])
    links.new(mp.outputs["Vector"],  wv.inputs["Vector"])
    links.new(wv.outputs["Fac"],     cr.inputs["Fac"])
    links.new(cr.outputs["Color"],   bsdf.inputs["Base Color"])

    mp.inputs["Scale"].default_value       = (rng.uniform(5, 20), 1.0, 1.0)
    wv.wave_type                           = "BANDS"
    wv.inputs["Scale"].default_value       = rng.uniform(30, 80)
    wv.inputs["Detail"].default_value      = rng.uniform(4,  10)
    wv.inputs["Distortion"].default_value  = rng.uniform(0.1, 0.5)

    palettes = [
        ((0.50, 0.51, 0.53), (0.75, 0.76, 0.78)),  # steel
        ((0.62, 0.52, 0.38), (0.82, 0.70, 0.52)),  # gold
        ((0.52, 0.32, 0.20), (0.78, 0.50, 0.30)),  # copper
        ((0.65, 0.67, 0.68), (0.88, 0.90, 0.92)),  # aluminum
    ]
    dark, light = rng.choice(palettes)
    cr.color_ramp.elements[0].color        = (*dark,  1.0)
    cr.color_ramp.elements[1].color        = (*light, 1.0)
    bsdf.inputs["Metallic"].default_value  = 1.0
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.05, 0.30)


def add_ground_plane(bbox, orbit_radius: float, rng: random.Random):
    """Large background plane with procedural wood / marble / metal texture."""
    import bpy
    min_z   = bbox[4]
    plane_z = min_z - 0.001 * orbit_radius

    plane_size = orbit_radius * 15.0
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(0.0, 0.0, plane_z))
    plane      = bpy.context.active_object
    plane.name = "Ground"

    mat   = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (600, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    choice = rng.random()
    if choice < 0.40:
        _ground_wood(nodes, links, bsdf, rng)
    elif choice < 0.70:
        _ground_marble(nodes, links, bsdf, rng)
    else:
        _ground_metal(nodes, links, bsdf, rng)

    plane.data.materials.append(mat)
    return plane


def add_background_blobs(bbox, orbit_radius: float, ground_z: float,
                         rng: random.Random):
    """Scatter colored icosphere blobs on the ground plane as background fill."""
    import bpy

    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    longest_dim = max(max_x - min_x, max_y - min_y, max_z - min_z)

    colors = [
        (0.82, 0.12, 0.12), (0.15, 0.42, 0.82), (0.12, 0.68, 0.22),
        (0.85, 0.65, 0.05), (0.62, 0.12, 0.72), (0.88, 0.38, 0.08),
        (0.08, 0.62, 0.62), (0.88, 0.22, 0.52), (0.25, 0.25, 0.25),
        (0.90, 0.88, 0.82),
    ]
    inner_r = longest_dim * 1.5
    outer_r = max(inner_r * 1.1, orbit_radius * 0.80)

    for i in range(rng.randint(5, 12)):
        angle  = rng.uniform(0, 2 * math.pi)
        dist   = rng.uniform(inner_r, outer_r)
        radius = rng.uniform(0.08 * longest_dim, 0.32 * longest_dim)
        bx     = dist * math.cos(angle)
        by     = dist * math.sin(angle)
        bz     = ground_z + radius * rng.uniform(0.6, 1.0)

        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=radius, subdivisions=3, location=(bx, by, bz)
        )
        blob      = bpy.context.active_object
        blob.name = f"Blob_{i}"
        blob.scale = (
            rng.uniform(0.7, 1.4),
            rng.uniform(0.7, 1.4),
            rng.uniform(0.4, 1.1),
        )
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        c   = rng.choice(colors)
        mat = bpy.data.materials.new(f"BlobMat_{i}")
        mat.use_nodes = True
        bn = mat.node_tree.nodes
        bl = mat.node_tree.links
        for n in list(bn):
            bn.remove(n)
        bsdf = bn.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0,   0)
        bout = bn.new("ShaderNodeOutputMaterial"); bout.location  = (300, 0)
        bl.new(bsdf.outputs["BSDF"], bout.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = (*c, 1.0)
        bsdf.inputs["Metallic"].default_value   = rng.uniform(0.0, 0.4)
        bsdf.inputs["Roughness"].default_value  = rng.uniform(0.2, 0.8)
        blob.data.materials.append(mat)


def apply_object_material(obj, rng: random.Random):
    """Replace the default STL grey with a randomised Principled BSDF."""
    import bpy
    mat   = bpy.data.materials.new("ObjectMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0, 0)
    out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if rng.random() < 0.30:                         # metallic — spec highlights
        metal_palette = [                           # keep as-is; sheen makes
            (0.72, 0.45, 0.20),  # copper / bronze # metals legible even on grey
            (0.83, 0.68, 0.21),  # gold
            (0.55, 0.57, 0.58),  # dark titanium
            (0.35, 0.35, 0.38),  # dark steel
        ]
        c = rng.choice(metal_palette)
        bsdf.inputs["Base Color"].default_value = (*c, 1.0)
        bsdf.inputs["Metallic"].default_value   = 1.0
        bsdf.inputs["Roughness"].default_value  = rng.uniform(0.05, 0.40)
    else:                                           # plastic / ceramic
        # Only saturated or dark colours — avoids blending with the neutral
        # light-grey / cream / beige ground plane tones.
        plastic_palette = [
            (0.12, 0.12, 0.12),  # near-black
            (0.18, 0.35, 0.72),  # cobalt blue
            (0.72, 0.12, 0.12),  # deep red
            (0.10, 0.52, 0.22),  # forest green
            (0.80, 0.55, 0.05),  # amber / orange
            (0.55, 0.15, 0.60),  # purple
            (0.08, 0.42, 0.52),  # teal
            (0.75, 0.30, 0.10),  # rust orange
        ]
        c = rng.choice(plastic_palette)
        bsdf.inputs["Base Color"].default_value = (*c, 1.0)
        bsdf.inputs["Metallic"].default_value   = 0.0
        bsdf.inputs["Roughness"].default_value  = rng.uniform(0.20, 0.75)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Camera rig and animation
# ---------------------------------------------------------------------------

def add_camera_rig(focal_mm: float, orbit_radius: float, elevation_deg: float,
                   pivot: tuple = (0.0, 0.0, 0.0)):
    """Orbit empty at *pivot* with a camera child.  Returns (camera, empty).

    *pivot* should be the midpoint between the object centre and the ArUco
    marker so both are framed symmetrically from every orbit azimuth.
    """
    import bpy
    import mathutils

    elev = math.radians(elevation_deg)

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=pivot)
    empty      = bpy.context.active_object
    empty.name = "OrbitEmpty"
    empty.rotation_euler = (0, 0, 0)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam_obj      = bpy.context.active_object
    cam_obj.name = "OrbitCam"

    cam_obj.parent = empty
    cam_obj.matrix_parent_inverse.identity()

    cam_x = orbit_radius * math.cos(elev)
    cam_z = orbit_radius * math.sin(elev)
    cam_obj.location = (cam_x, 0.0, cam_z)

    # Camera looks at the pivot (local origin of the orbit empty)
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((cam_x, 0.0, cam_z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    cam_obj.data.lens        = focal_mm
    cam_obj.data.sensor_width = 36.0
    # Tight near/far ratio for depth-buffer precision. Scenes max out around
    # a few metres; an extreme 1e-4 / 1e7 range causes z-fighting.
    cam_obj.data.clip_start  = 0.005
    cam_obj.data.clip_end    = orbit_radius * 20.0

    bpy.context.scene.camera = cam_obj
    return cam_obj, empty


def animate_orbit(orbit_empty, num_frames: int):
    """Seamless 360° orbit: frame 1 → 0°, frame N → 360°·(N-1)/N, LINEAR."""
    import bpy
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = num_frames
    end_angle = math.radians(360.0 * (num_frames - 1) / num_frames)

    scene.frame_set(1)
    orbit_empty.rotation_euler = (0, 0, 0)
    orbit_empty.keyframe_insert(data_path="rotation_euler", index=2)

    scene.frame_set(num_frames)
    orbit_empty.rotation_euler = (0, 0, end_angle)
    orbit_empty.keyframe_insert(data_path="rotation_euler", index=2)

    for fc in orbit_empty.animation_data.action.fcurves:
        if fc.data_path == "rotation_euler" and fc.array_index == 2:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    scene.frame_set(1)


def apply_shake(scene, cam_obj, num_frames: int,
                orbit_radius: float, rng: random.Random):
    """Insert per-frame keyframes with Perlin-noise position and rotation jitter.

    cameras.json is collected AFTER this call so it reflects the actual
    perturbed poses used during rendering.
    """
    import mathutils

    jitter_pos = orbit_radius * 0.015   # 1.5 % of orbit radius
    jitter_rot = math.radians(0.5)       # 0.5° max per axis
    freq       = 0.07                    # ~1.7 Hz at 24fps → smooth hand-sway

    phases = [rng.uniform(0, 100) for _ in range(6)]
    base_loc = cam_obj.location.copy()
    base_rot = cam_obj.rotation_euler.copy()

    for frame in range(1, num_frames + 1):
        t = frame * freq

        def pn(phase):
            return mathutils.noise.noise(mathutils.Vector((t, phase, 0.0)))

        dx  = pn(phases[0]) * jitter_pos
        dy  = pn(phases[1]) * jitter_pos
        dz  = pn(phases[2]) * jitter_pos * 0.3   # less vertical bounce
        drx = pn(phases[3]) * jitter_rot
        dry = pn(phases[4]) * jitter_rot
        drz = pn(phases[5]) * jitter_rot * 0.5   # less roll

        cam_obj.location       = (base_loc.x + dx, base_loc.y + dy, base_loc.z + dz)
        cam_obj.rotation_euler = (base_rot.x + drx, base_rot.y + dry, base_rot.z + drz)

        cam_obj.keyframe_insert(data_path="location",       frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Constant interpolation — each frame holds its own independent pose
    if cam_obj.animation_data and cam_obj.animation_data.action:
        for fc in cam_obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    scene.frame_set(1)


# ---------------------------------------------------------------------------
# Render & compositor
# ---------------------------------------------------------------------------

def configure_render(scene, output_dir: Path, depth_dir: Path, mask_dir: Path,
                     num_frames: int, fps: int, resolution: list, engine: str,
                     no_exr: bool = False):
    import bpy
    scene.render.engine             = engine
    scene.render.resolution_x       = resolution[0]
    scene.render.resolution_y       = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.fps                = fps
    scene.render.use_persistent_data = True
    view_layer = scene.view_layers[0]

    scene.render.filepath                      = str(output_dir / "video")
    scene.render.image_settings.file_format    = "FFMPEG"
    scene.render.ffmpeg.format                 = "MPEG4"
    scene.render.ffmpeg.codec                  = "H264"
    scene.render.ffmpeg.constant_rate_factor   = "MEDIUM"
    scene.render.ffmpeg.audio_codec            = "NONE"

    if no_exr:
        # VLM-only render: video + cameras.json, no per-frame EXR passes.
        scene.use_nodes = False
        return

    view_layer.use_pass_z                = True
    # Cryptomatte object pass → per-object coverage mask via the matching
    # CompositorNodeCryptomatteV2 below. EEVEE's IndexOB pass is documented
    # but does not actually populate output in Blender 3.6.
    view_layer.use_pass_cryptomatte_object = True

    scene.use_nodes = True
    tree  = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    for n in list(nodes):
        nodes.remove(n)

    rl = nodes.new("CompositorNodeRLayers");       rl.location  = (0,   0)
    cp = nodes.new("CompositorNodeComposite");     cp.location  = (400, 250)
    fo = nodes.new("CompositorNodeOutputFile");    fo.location  = (400, 0)
    cm = nodes.new("CompositorNodeCryptomatteV2"); cm.location  = (400, -200)
    mo = nodes.new("CompositorNodeOutputFile");    mo.location  = (700, -200)

    links.new(rl.outputs["Image"], cp.inputs["Image"])
    links.new(rl.outputs["Depth"], fo.inputs[0])

    fo.base_path                    = str(depth_dir) + os.sep
    fo.format.file_format           = "OPEN_EXR"
    fo.format.color_depth           = "32"
    fo.file_slots[0].path           = "frame_"

    # Per-pixel object mask via Cryptomatte object pass. Matte output is a
    # float coverage map (1 inside the CAD object, 0 elsewhere); training
    # loss applies only where mask > 0 so background blobs are ignored.
    cm.matte_id = "CADObject"
    mo.base_path                    = str(mask_dir) + os.sep
    mo.format.file_format           = "OPEN_EXR"
    mo.format.color_depth           = "16"
    mo.file_slots[0].path           = "frame_"
    links.new(cm.outputs["Matte"], mo.inputs[0])


# ---------------------------------------------------------------------------
# Camera data collection
# ---------------------------------------------------------------------------

def collect_cameras(scene, cam_obj, num_frames: int, resolution: list) -> list:
    """Per-frame camera intrinsics + extrinsics (OpenCV convention).
    Call this AFTER apply_shake so the perturbed poses are captured.
    """
    import bpy
    import mathutils

    W, H      = resolution
    focal_mm  = cam_obj.data.lens
    sensor_w  = cam_obj.data.sensor_width
    fx = fy   = focal_mm / sensor_w * W
    cx, cy    = W / 2.0, H / 2.0

    flip = mathutils.Matrix(((1, 0, 0, 0),
                             (0,-1, 0, 0),
                             (0, 0,-1, 0),
                             (0, 0, 0, 1)))
    records = []
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam_eval  = cam_obj.evaluated_get(depsgraph)
        c2w       = cam_eval.matrix_world.copy()
        pos       = list(c2w.translation)
        w2c       = flip @ c2w.inverted()
        records.append({
            "frame":           frame,
            "position_mm":     [pos[0], pos[1], pos[2]],
            "world_to_camera": [[w2c[r][c] for c in range(4)] for r in range(4)],
            "convention":      "opencv",
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": W, "height": H,
        })
    scene.frame_set(1)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import bpy

    args = parse_args()
    output_dir = args.output_dir
    depth_dir  = output_dir / "depth"
    mask_dir   = output_dir / "mask"
    if not args.no_exr:
        depth_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic per-object RNG
    seed = derive_seed(output_dir)
    rng  = random.Random(seed)

    stl_path   = args.stl_path.resolve()
    aruco_path = args.aruco_marker.resolve()
    print(f"[render_scene] stl={stl_path}  out={output_dir}  seed={seed}")

    # --- Build scene ---
    clear_scene()

    obj = import_stl(stl_path)
    center_object(obj)
    apply_object_material(obj, rng)

    bbox        = world_bbox(obj)
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    dims        = (max_x - min_x, max_y - min_y, max_z - min_z)
    longest_dim = max(dims)
    orbit_radius = args.orbit_radius_factor * longest_dim

    # Randomise camera elevation (centre ± 20°, clamped 10°–70°)
    elev_center  = args.elevation_deg
    elevation_deg = rng.uniform(
        max(10.0, elev_center - 20.0),
        min(70.0, elev_center + 20.0),
    )

    ground_z = min_z - 0.001 * orbit_radius
    # On a 1.5 m object the default 50 mm marker projects to ~8 px and is
    # undetectable. Floor the physical size at 15 % of the longest dimension
    # so it always renders at >30 px regardless of object scale.
    longest_dim_mm  = longest_dim * SCENE_UNIT_MM
    effective_aruco_size_mm = max(args.aruco_size_mm, longest_dim_mm * 0.15)
    add_aruco_plane(aruco_path, bbox, effective_aruco_size_mm, longest_dim, ground_z, orbit_radius)
    add_ground_plane(bbox, orbit_radius, rng)
    if args.enable_blobs:
        add_background_blobs(bbox, orbit_radius, ground_z, rng)
    setup_world_sky(rng)
    setup_lighting(rng)

    # Pivot at the midpoint between the object centre and the ArUco marker
    # centre so both are symmetrically framed from every orbit azimuth.
    aruco_size_scene = effective_aruco_size_mm / SCENE_UNIT_MM
    aruco_gap        = 0.2 * longest_dim
    marker_cx        = max_x + aruco_gap + aruco_size_scene * 0.5
    pivot            = (marker_cx / 2.0, 0.0, 0.0)

    cam_obj, orbit_empty = add_camera_rig(
        args.focal_length_mm, orbit_radius, elevation_deg, pivot=pivot
    )
    animate_orbit(orbit_empty, args.num_frames)
    apply_shake(bpy.context.scene, cam_obj, args.num_frames, orbit_radius, rng)

    scene = bpy.context.scene
    configure_render(scene, output_dir, depth_dir, mask_dir,
                     args.num_frames, args.fps, args.resolution, args.engine,
                     no_exr=args.no_exr)

    # Collect camera poses AFTER shake keyframes are in place
    cameras = collect_cameras(scene, cam_obj, args.num_frames, args.resolution)
    (output_dir / "cameras.json").write_text(json.dumps(cameras, indent=2))

    aruco_center_mm = [marker_cx, 0.0, ground_z + 0.05 * orbit_radius]
    metadata = {
        "num_frames":        args.num_frames,
        "resolution":        args.resolution,
        "bbox_min_mm":       [min_x, min_y, min_z],
        "bbox_max_mm":       [max_x, max_y, max_z],
        "bbox_dims_mm":      list(dims),
        "longest_dim_mm":    longest_dim,
        "orbit_radius_mm":   orbit_radius,
        "elevation_deg":     elevation_deg,
        "render_seed":       seed,
        "aruco_size_mm":     effective_aruco_size_mm,
        "aruco_center_mm":   aruco_center_mm,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[render_scene] Rendering {args.num_frames} frames …")
    bpy.ops.render.render(animation=True)

    # Rename Blender's frame-stamped output (e.g. video0001-0060.mp4) → video.mp4
    for candidate in output_dir.glob("video*.mp4"):
        if candidate.name != "video.mp4":
            candidate.rename(output_dir / "video.mp4")
            break

    print(f"[render_scene] Done → {output_dir}")


if __name__ == "__main__":
    main()
