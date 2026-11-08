"""
Headless tests for Emil's Normal Brush. Run with Blender 3.6:

    blender -b --factory-startup --python EmilsNormalBrush/tests/run_tests.py -- <result.txt>

Results go to <result.txt>, prints and tracebacks to <result.txt>.log.
"""

import math
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
RESULT = argv[0] if argv else os.path.join(tempfile.gettempdir(), "emils_normal_brush_tests.txt")
_log = open(RESULT + ".log", "w", encoding="utf-8", buffering=1)
sys.stdout = _log
sys.stderr = _log
import faulthandler  # noqa: E402
faulthandler.enable(_log)

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

import EmilsNormalBrush  # noqa: E402
from EmilsNormalBrush import brush, ui  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print(results[-1], flush=True)


def info(text):
    results.append("INFO  " + text)
    print(results[-1], flush=True)


def make_balls(centers, segments=32, rings=16, smooth=True):
    """One mesh object made of separate (unwelded) UV spheres, one per center."""
    bm = bmesh.new()
    for c in centers:
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0,
                                  matrix=Matrix.Translation(c))
    mesh = bpy.data.meshes.new("Balls")
    bm.to_mesh(mesh)
    per = len(bm.verts) // len(centers)
    bm.free()
    if smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj = bpy.data.objects.new("Balls", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj, per


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


def seam_angle(session, per, a, b, near=0.15):
    """Mean angle (deg) between each sphere-a vertex near sphere b and the closest sphere-b vertex."""
    vw = session.vw
    vn = session.state.vn
    pa, pb = vw[a * per:(a + 1) * per], vw[b * per:(b + 1) * per]
    na, nb = vn[a * per:(a + 1) * per], vn[b * per:(b + 1) * per]
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    j = d.argmin(axis=1)
    sel = d[np.arange(len(pa)), j] < near
    dots = np.clip(np.einsum("ij,ij->i", na[sel], nb[j[sel]]), -1, 1)
    return float(np.degrees(np.arccos(dots)).mean()), int(sel.sum())


def seam_ring(center_x, count=12, y=0.8):
    """Points on the intersection circle of two unit balls whose intersection plane is x = center_x."""
    r = math.sqrt(max(1.0 - 0.36, 0.0))
    return [(center_x, r * math.cos(t), r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, count, endpoint=False)]


class Settings:
    strength = 0.8
    blur_distance = 0.3
    mirror_x = False


def run():
    scene = bpy.context.scene
    EmilsNormalBrush.register()
    clear_scene()

    # ---- seam blur
    obj, per = make_balls([(0, 0, 0), (1.2, 0, 0)])
    s = Settings()
    sess = brush.Session(obj)
    before, count = seam_angle(sess, per, 0, 1)
    ring = seam_ring(0.6)
    # a dab far from the seam, to check nothing outside the radius moves
    start = sess.state.ln.copy()
    sess.dab((0.6, 0.8, 0.0), 0.3, s)
    moved = np.nonzero(np.linalg.norm(sess.state.ln - start, axis=1) > 1e-7)[0]
    vd = np.linalg.norm(sess.vw[sess.state.loop_vert[moved]] - np.array([0.6, 0.8, 0.0]), axis=1)
    check("one dab only moves loops inside the brush radius", len(moved) > 0 and vd.max() < 0.3 + 1e-4,
          f"{len(moved)} loops moved, farthest {vd.max():.3f}")
    far = np.linalg.norm(sess.vw[sess.state.loop_vert] - np.array([0.6, 0.8, 0.0]), axis=1) > 0.3
    check("loops outside the radius are bit-identical", np.array_equal(sess.state.ln[far], start[far]))
    for _ in range(4):
        for p in ring:
            sess.dab(p, 0.5, s)
    after, _ = seam_angle(sess, per, 0, 1)
    info(f"seam angle before {before:.1f} deg, after {after:.1f} deg ({count} seam vertices)")
    check("blur dabs bring the two islands' seam normals together", after < before * 0.5,
          f"{before:.1f} -> {after:.1f} deg")
    sess.write()
    check("mesh.has_custom_normals after the write", obj.data.has_custom_normals)
    obj.data.calc_normals_split()
    back = np.empty(len(obj.data.loops) * 3, dtype=np.float32)
    obj.data.loops.foreach_get("normal", back)
    err = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", back.reshape(-1, 3), sess.state.ln), -1, 1))).max()
    check("normals written to the mesh match the numpy state", err < 1.0, f"max {err:.3f} deg")

    # ---- restore
    plain = sess.state.plain
    def off_plain():
        dots = np.clip(np.einsum("ij,ij->i", sess.state.vn, plain), -1, 1)
        return float(np.degrees(np.arccos(dots)).mean())
    blurred_off = off_plain()
    for _ in range(4):
        for p in ring:
            sess.dab(p, 0.6, s, "RESTORE")
    for _ in range(2):
        sess.dab((0.6, 0.0, 0.8), 0.9, Settings(), "RESTORE")
    restored_off = off_plain()
    info(f"mean angle to plain vertex normals: blurred {blurred_off:.2f} deg, restored {restored_off:.2f} deg")
    check("restore dabs bring normals back toward the plain vertex normals", restored_off < blurred_off * 0.5,
          f"{blurred_off:.2f} -> {restored_off:.2f} deg")
    seam_after_restore, _ = seam_angle(sess, per, 0, 1)
    check("restore undoes the seam blur too", seam_after_restore > after, f"{after:.1f} -> {seam_after_restore:.1f}")

    # ---- cancel (restore_start)
    sess.restore_start()
    check("restore_start gives back the start normals", np.array_equal(sess.state.ln, sess.state.start_ln))

    # ---- clear operator + modifier warning
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.emils.normal_brush_clear()
    check("Clear Custom Normals removes them", not obj.data.has_custom_normals)
    mod = obj.modifiers.new("WN", "WEIGHTED_NORMAL")
    check("modifier warning sees Weighted Normal", ui.overwriting_modifiers(obj) == ["WN"])
    obj.modifiers.remove(mod)
    dt = obj.modifiers.new("DT", "DATA_TRANSFER")
    check("Data Transfer without custom normals isn't flagged", ui.overwriting_modifiers(obj) == [])
    dt.use_loop_data = True
    dt.data_types_loops = {"CUSTOM_NORMAL"}
    check("Data Transfer with custom normals is flagged", ui.overwriting_modifiers(obj) == ["DT"])
    obj.modifiers.remove(dt)

    # ---- auto distance
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.emils.normal_brush_auto_distance()
    d = scene.emils_normal_brush.blur_distance
    mean_edge = brush.core.mean_edge_length(brush.world_positions(obj), brush.mesh_edges(obj.data))
    check("Auto sets blur distance to 2.5 x mean edge length", abs(d - 2.5 * mean_edge) < 1e-5 and d > 0.05,
          f"{d:.4f}")

    # ---- mirror
    clear_scene()
    obj, per = make_balls([(2.0, 0, 0), (3.2, 0, 0), (-2.0, 0, 0), (-3.2, 0, 0)])
    s = Settings()
    s.mirror_x = True
    sess = brush.Session(obj)
    r_before, _ = seam_angle(sess, per, 0, 1)
    l_before, _ = seam_angle(sess, per, 2, 3)
    for _ in range(4):
        for p in seam_ring(2.6):
            sess.dab(p, 0.5, s)
    r_after, _ = seam_angle(sess, per, 0, 1)
    l_after, _ = seam_angle(sess, per, 2, 3)
    info(f"mirror: right seam {r_before:.1f} -> {r_after:.1f}, left seam {l_before:.1f} -> {l_after:.1f}")
    check("mirror_x dabs both sides", r_after < r_before * 0.5 and l_after < l_before * 0.5)
    s.mirror_x = False
    sess2 = brush.Session(make_balls([(2.0, 0, 0), (3.2, 0, 0), (-2.0, 0, 0), (-3.2, 0, 0)])[0])
    for p in seam_ring(2.6):
        sess2.dab(p, 0.5, s)
    check("without mirror_x the other side stays put", seam_angle(sess2, per, 2, 3)[0] > 0.99 * l_before)

    # ---- auto smooth
    clear_scene()
    obj, per = make_balls([(0, 0, 0)])
    check("auto smooth starts off", not obj.data.use_auto_smooth)
    brush.Session(obj)
    check("Session turns auto smooth on with angle pi",
          obj.data.use_auto_smooth and abs(obj.data.auto_smooth_angle - math.pi) < 1e-4,
          f"{obj.data.auto_smooth_angle:.4f}")
    obj.data.auto_smooth_angle = 0.5
    brush.Session(obj)
    check("auto smooth already on keeps the user's angle",
          obj.data.use_auto_smooth and abs(obj.data.auto_smooth_angle - 0.5) < 1e-4)

    # ---- timing on a ~20k loop mesh
    clear_scene()
    obj, per = make_balls([(0, 0, 0)], segments=100, rings=50)
    nloops = len(obj.data.loops)
    t0 = time.perf_counter()
    sess = brush.Session(obj)
    t_setup = (time.perf_counter() - t0) * 1000
    s = Settings()
    s.strength = 0.5
    s.blur_distance = brush.auto_blur_distance(obj)
    for radius in (0.35, 0.8):
        dab_t, write_t = [], []
        for i in range(20):
            a = i * 0.3
            hit = (math.cos(a) * 0.7, math.sin(a) * 0.7, math.sqrt(1 - 0.49))
            t0 = time.perf_counter()
            sess.dab(hit, radius, s)
            dab_t.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            sess.write()
            write_t.append((time.perf_counter() - t0) * 1000)
        aff = int((np.linalg.norm(sess.vw - np.array([0.7, 0.0, 0.714]), axis=1) < radius).sum())
        total = np.median(dab_t) + np.median(write_t)
        info(f"timing on {nloops} loops, brush radius {radius} (~{aff} verts): setup {t_setup:.0f} ms, "
             f"dab median {np.median(dab_t):.1f} ms (max {max(dab_t):.1f}), "
             f"write median {np.median(write_t):.1f} ms (max {max(write_t):.1f})")
        # a normal brush (~150 verts) has to fit 30 ms, a huge one (a third of the ball) gets a looser limit,
        # the strokes are throttled to 30 writes a second anyway
        limit = 30 if radius < 0.5 else 60
        check(f"dab + write under {limit} ms on ~20k loops (radius {radius})", total < limit, f"{total:.1f} ms")
    clear_scene()

    # ---- register / unregister twice
    for i in range(2):
        EmilsNormalBrush.unregister()
        check(f"unregister #{i + 1} removes the scene property", not hasattr(bpy.types.Scene, "emils_normal_brush"))
        EmilsNormalBrush.register()
        check(f"register #{i + 1} brings props, panel and operators back",
              hasattr(bpy.types.Scene, "emils_normal_brush") and hasattr(bpy.types, "EMILSNB_PT_main")
              and hasattr(bpy.ops.emils, "normal_brush_stroke"))
    EmilsNormalBrush.unregister()
    check("final unregister", not hasattr(bpy.types, "EMILSNB_PT_main"))


try:
    run()
except Exception:
    traceback.print_exc()
    results.append("FAIL  crashed: " + traceback.format_exc().splitlines()[-1])
with open(RESULT, "w", encoding="utf-8") as f:
    f.write("\n".join(results) + "\n")
os._exit(0)
