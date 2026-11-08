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
from mathutils import Matrix, Vector  # noqa: E402

import EmilsNormalBrush  # noqa: E402
from EmilsNormalBrush import brush, ui  # noqa: E402
from EmilsNormalBrush.core import normalize as core_normalize  # noqa: E402

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


def make_two_balls(segments=32, rings=16):
    """Two SEPARATE ball objects, overlapping. The second one is rotated around an odd axis, scaled 1.3, moved."""
    clear_scene()
    a, per_a = make_balls([(0, 0, 0)], segments, rings)
    a.name = "BallA"
    b, per_b = make_balls([(0, 0, 0)], segments, rings)
    b.name = "BallB"
    b.matrix_world = (Matrix.Translation((1.5, 0.3, 0.2))
                      @ Matrix.Rotation(0.7, 4, Vector((1.0, 2.0, 3.0)).normalized()) @ Matrix.Scale(1.3, 4))
    bpy.context.view_layer.update()
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = a
    return a, b


def world_vn(sess, i):
    s = sess.state
    return s.vn[s.offsets[i]:s.offsets[i + 1]], sess.vw[s.offsets[i]:s.offsets[i + 1]]


def cross_seam(sess, i=0, j=1, near=0.2):
    """Seam vertices of object i (those within `near` of an object-j vertex), their indices in the session's
    concatenated arrays, and the mean world-space angle (deg) to the closest object-j vertex normal."""
    na, pa = world_vn(sess, i)
    nb, pb = world_vn(sess, j)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    k = d.argmin(axis=1)
    sel = np.nonzero(d[np.arange(len(pa)), k] < near)[0]
    dots = np.clip(np.einsum("ij,ij->i", na[sel], nb[k[sel]]), -1, 1)
    return float(np.degrees(np.arccos(dots)).mean()), sel


def read_loop_normals(mesh):
    mesh.calc_normals_split()
    a = np.empty(len(mesh.loops) * 3, dtype=np.float32)
    mesh.loops.foreach_get("normal", a)
    return a.reshape(-1, 3)


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
    start = sess.state.objs[0].ln.copy()
    sess.dab((0.6, 0.8, 0.0), 0.3, s)
    moved = np.nonzero(np.linalg.norm(sess.state.objs[0].ln - start, axis=1) > 1e-7)[0]
    vd = np.linalg.norm(sess.vw[sess.state.objs[0].loop_vert[moved]] - np.array([0.6, 0.8, 0.0]), axis=1)
    check("one dab only moves loops inside the brush radius", len(moved) > 0 and vd.max() < 0.3 + 1e-4,
          f"{len(moved)} loops moved, farthest {vd.max():.3f}")
    far = np.linalg.norm(sess.vw[sess.state.objs[0].loop_vert] - np.array([0.6, 0.8, 0.0]), axis=1) > 0.3
    check("loops outside the radius are bit-identical", np.array_equal(sess.state.objs[0].ln[far], start[far]))
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
    err = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", back.reshape(-1, 3), sess.state.objs[0].ln), -1, 1))).max()
    check("normals written to the mesh match the numpy state", err < 1.0, f"max {err:.3f} deg")

    # ---- restore
    plain = sess.state.objs[0].plain
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
    check("restore_start gives back the start normals", np.array_equal(sess.state.objs[0].ln, sess.state.objs[0].start_ln))

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

    # ---- two separate objects, one with a nasty transform
    a, b = make_two_balls()
    s = Settings()
    sess = brush.Session([a, b], active=a)
    check("session paints both objects", len(sess.objects) == 2 and len(sess.state.objs) == 2)
    before, seam = cross_seam(sess)
    seam_pts = sess.vw[seam]            # object A is first in the concatenation, so its indices are the same
    na0, pa0 = world_vn(sess, 0)
    nb0, pb0 = world_vn(sess, 1)
    start = [o.ln.copy() for o in sess.state.objs]
    start_vn = sess.state.vn.copy()
    for _ in range(4):
        for p in seam_pts:
            sess.dab(tuple(p), 0.5, s)
    after, _ = cross_seam(sess)
    info(f"two objects: seam angle across objects before {before:.1f} deg, after {after:.1f} deg "
         f"({len(seam)} seam vertices)")
    check("two objects: blur brings the seam normals of both objects together", after < before * 0.4,
          f"{before:.1f} -> {after:.1f} deg")
    dist_to_dabs = np.linalg.norm(sess.vw[:, None, :] - seam_pts[None, :, :], axis=2).min(axis=1)
    far_v = dist_to_dabs > 0.5 + 1e-4
    check("two objects: vertices far from every dab keep their normals",
          far_v.sum() > 100 and np.allclose(sess.state.vn[far_v], start_vn[far_v], atol=1e-6),
          f"{int(far_v.sum())} vertices far away")
    for i, o in enumerate(sess.state.objs):
        far_l = far_v[sess.state.offsets[i]:sess.state.offsets[i + 1]][o.loop_vert]
        check(f"two objects: object {i} loops far from the dabs are bit-identical",
              np.array_equal(o.ln[far_l], start[i][far_l]), f"{int(far_l.sum())} loops")
        check(f"two objects: object {i} actually changed", not np.array_equal(o.ln, start[i]))
    sess.write()
    check("two objects: both meshes have custom normals", a.data.has_custom_normals and b.data.has_custom_normals)
    # read the normals back from the meshes, to world space with each object's own matrix
    worst = 0.0
    for i, ob in enumerate((a, b)):
        back = read_loop_normals(ob.data)
        o = sess.state.objs[i]
        err = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", back, o.ln), -1, 1))).max()
        worst = max(worst, err)
        m3 = np.array(ob.matrix_world.to_3x3(), dtype=np.float64)
        wn = core_normalize(back @ np.linalg.inv(m3))
        sums = np.stack([np.bincount(o.loop_vert, weights=wn[:, k], minlength=o.nverts) for k in range(3)], axis=1)
        vn_back = core_normalize(sums)
        err_w = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", vn_back, o.vn_world), -1, 1))).max()
        worst = max(worst, err_w)
    check("two objects: normals written to both meshes match the state in local and world space", worst < 1.0,
          f"max {worst:.3f} deg")

    # restore across both objects
    def off_plain():
        out = []
        for i, o in enumerate(sess.state.objs):
            plain_w = core_normalize(o.plain @ o.to_world.T)
            vn, _ = world_vn(sess, i)
            near_dab = dist_to_dabs[sess.state.offsets[i]:sess.state.offsets[i + 1]] < 0.5
            dots = np.clip(np.einsum("ij,ij->i", vn[near_dab], plain_w[near_dab]), -1, 1)
            out.append(float(np.degrees(np.arccos(dots)).mean()))
        return out
    blurred_off = off_plain()
    r = Settings()
    r.strength = 1.0
    for _ in range(5):
        for p in seam_pts:
            sess.dab(tuple(p), 0.6, r, "RESTORE")
    restored_off = off_plain()
    info(f"two objects: mean angle to plain normals near the seam, blurred {blurred_off[0]:.2f}/{blurred_off[1]:.2f}"
         f" deg, restored {restored_off[0]:.3f}/{restored_off[1]:.3f} deg")
    check("two objects: restore brings both back to their plain normals",
          all(rest < 0.15 * bl + 0.05 for rest, bl in zip(restored_off, blurred_off)) and min(blurred_off) > 0.5,
          f"{blurred_off} -> {restored_off}")
    sess.restore_start()
    check("two objects: restore_start gives every object its start normals back",
          all(np.array_equal(o.ln, st) for o, st in zip(sess.state.objs, start)))

    # only selected meshes
    a, b = make_two_balls()
    c, _ = make_balls([(0, 0, 0)])
    c.name = "BallC"
    c.matrix_world = Matrix.Translation((0.7, -0.4, 0.3))
    c.select_set(False)
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = a
    bpy.context.view_layer.update()
    picked = brush.selected_meshes(bpy.context)
    check("selected_meshes gives the active mesh first and skips unselected ones",
          [ob.name for ob in picked] == ["BallA", "BallB"], str([ob.name for ob in picked]))
    c_before = read_loop_normals(c.data)
    c_smooth = c.data.use_auto_smooth
    sess = brush.Session(picked, active=bpy.context.active_object)
    for p in seam_ring(0.7, 8):
        sess.dab(p, 0.5, s)
    sess.write()
    check("unselected overlapping object keeps its normals bit-identical",
          np.array_equal(read_loop_normals(c.data), c_before) and not c.data.has_custom_normals
          and c.data.use_auto_smooth == c_smooth)

    # linked duplicate
    a, b = make_two_balls()
    dup = bpy.data.objects.new("DupOfA", a.data)
    bpy.context.scene.collection.objects.link(dup)
    dup.matrix_world = Matrix.Translation((0.4, 0.2, -0.1))
    dup.select_set(True)
    bpy.context.view_layer.objects.active = a
    picked = brush.selected_meshes(bpy.context)
    check("linked duplicates count once", len(picked) == 3 - 1 and sum(ob.data is a.data for ob in picked) == 1,
          str([ob.name for ob in picked]))
    sess = brush.Session([a, dup, b], active=a)
    check("linked duplicate gets skipped by the session", [ob.name for ob in sess.objects] == ["BallA", "BallB"])
    pts = [tuple(p) for p in sess.vw[cross_seam(sess)[1]]]
    for p in pts:
        sess.dab(p, 0.5, s)
    sess.write()
    got = [o.ln.copy() for o in sess.state.objs]
    sess.restore_start()
    sess2 = brush.Session([a, b], active=a)
    for p in pts:
        sess2.dab(p, 0.5, s)
    check("linked duplicate selected twice doesn't double-apply",
          all(np.allclose(g, o.ln, atol=1e-6) for g, o in zip(got, sess2.state.objs)))

    # operators over several selected meshes
    a, b = make_two_balls()
    with bpy.context.temp_override(active_object=a, object=a):
        bpy.ops.emils.normal_brush_auto_distance()
    d = scene.emils_normal_brush.blur_distance
    ea = brush.mesh_edges(a.data)
    lens = []
    for ob in (a, b):
        e, wp = brush.mesh_edges(ob.data), brush.world_positions(ob)
        lens.append(np.linalg.norm(wp[e[:, 0]] - wp[e[:, 1]], axis=1))
    expect = 2.5 * float(np.concatenate(lens).mean())
    only_a = 2.5 * float(lens[0].mean())
    check("Auto blur distance uses the mean edge length of all selected meshes",
          abs(d - expect) < 1e-5 and abs(d - only_a) > 1e-3, f"{d:.4f} (all) vs {only_a:.4f} (active only)")
    sess = brush.Session([a, b], active=a)
    sess.dab((0.7, 0.5, 0.2), 0.8, s)
    sess.write()
    check("both meshes have custom normals before clearing", a.data.has_custom_normals and b.data.has_custom_normals)
    with bpy.context.temp_override(active_object=a, object=a):
        bpy.ops.emils.normal_brush_clear()
    check("Clear Custom Normals clears every selected mesh",
          not a.data.has_custom_normals and not b.data.has_custom_normals)
    a.modifiers.new("WN", "WEIGHTED_NORMAL")
    b.modifiers.new("NE", "NORMAL_EDIT")
    check("modifier warning lists Object: Modifier over the selected meshes",
          ui.overwriting_modifiers_all([a, b]) == ["BallA: WN", "BallB: NE"], str(ui.overwriting_modifiers_all([a, b])))

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

    # ---- timing, two objects of ~10k loops each
    a, b = make_two_balls(segments=70, rings=35)
    nl = len(a.data.loops) + len(b.data.loops)
    t0 = time.perf_counter()
    sess = brush.Session([a, b], active=a)
    t_setup = (time.perf_counter() - t0) * 1000
    s = Settings()
    s.strength = 0.5
    s.blur_distance = brush.auto_blur_distance([a, b])
    seam_c = sess.vw[cross_seam(sess, near=0.1)[1]]
    for radius in (0.35, 0.8):
        dab_t, write_t = [], []
        for i in range(20):
            hit = tuple(seam_c[(i * 7) % len(seam_c)])
            t0 = time.perf_counter()
            sess.dab(hit, radius, s)
            dab_t.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            sess.write()
            write_t.append((time.perf_counter() - t0) * 1000)
        total = np.median(dab_t) + np.median(write_t)
        info(f"timing on 2 objects / {nl} loops, brush radius {radius}: setup {t_setup:.0f} ms, "
             f"dab median {np.median(dab_t):.1f} ms (max {max(dab_t):.1f}), "
             f"write median {np.median(write_t):.1f} ms (max {max(write_t):.1f})")
        limit = 30 if radius < 0.5 else 60
        check(f"2 objects: dab + write under {limit} ms on ~{nl // 1000}k loops (radius {radius})", total < limit,
              f"{total:.1f} ms")
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
