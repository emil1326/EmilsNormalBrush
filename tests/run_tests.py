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
    check("Auto sets blur distance to AUTO_FACTOR x mean edge length", abs(d - brush.AUTO_FACTOR * mean_edge) < 1e-5 and d > 0.05,
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

    # ---- low-poly: two coarse spheres as separate objects, one with a nasty transform
    def low_poly_pair(segments=8, rings=6):
        clear_scene()
        a, _ = make_balls([(0, 0, 0)], segments, rings)
        a.name = "LowA"
        b, _ = make_balls([(0, 0, 0)], segments, rings)
        b.name = "LowB"
        b.matrix_world = (Matrix.Translation((1.2, 0.2, 0.1))
                          @ Matrix.Rotation(0.5, 4, Vector((1.0, 2.0, 3.0)).normalized()) @ Matrix.Scale(1.2, 4))
        bpy.context.view_layer.update()
        a.select_set(True)
        b.select_set(True)
        bpy.context.view_layer.objects.active = a
        return a, b

    def old_vertex_targets(state):
        """The 1.1 blur target, kept here only to tell the difference: mean of the current vertex normals in reach."""
        def targets(aff, rel, hit, radius, blur_distance):
            d = np.linalg.norm(state.vw[aff][:, None, :] - state.vw[None, :, :], axis=2)
            out = (d < blur_distance).astype(np.float32) @ state.vn
            return core_normalize(out), np.ones(len(aff), dtype=bool)
        return targets

    def low_run(old, bd, label=None):
        a, b = low_poly_pair()
        sess = brush.Session([a, b], active=a)
        if old:
            sess.state._blur_targets = old_vertex_targets(sess.state)
        st = Settings()
        st.blur_distance = bd
        before, seam = cross_seam(sess, near=0.8)
        pts = [tuple(p) for p in sess.vw[seam]]
        nb, pb = world_vn(sess, 1)
        d = np.linalg.norm(sess.vw[seam][:, None, :] - pb[None, :, :], axis=2)
        pts += [tuple(pb[j]) for j in np.unique(d.argmin(axis=1))]
        for _ in range(4):
            for p in pts:
                sess.dab(p, 0.9, st)
        after, _ = cross_seam(sess, near=0.8)
        return before, after, len(seam)

    for f in (0.5, 1.0, 1.5, 2.0, 2.5):
        a, b = low_poly_pair()
        e = brush.core.mean_edge_length(brush.world_positions(a), brush.mesh_edges(a.data))
        bd = f * e
        nb_, na_, n_ = low_run(False, bd)
        ob_, oa_, _ = low_run(True, bd)
        info(f"low-poly factor {f} (blur distance {bd:.3f}, mean edge {e:.3f}): faces {nb_:.1f} -> {na_:.1f} deg, "
             f"old vertex method {ob_:.1f} -> {oa_:.1f} deg ({n_} seam vertices)")
    a, b = low_poly_pair()
    bd = brush.auto_blur_distance([a, b])
    nb_, na_, n_ = low_run(False, bd)
    ob_, oa_, _ = low_run(True, bd)
    info(f"low-poly with Auto distance {bd:.3f}: faces {nb_:.1f} -> {na_:.1f} deg, old vertex method {ob_:.1f} -> {oa_:.1f} deg")
    check("low-poly: face blur brings the seam of two coarse spheres together", na_ < nb_ * 0.4, f"{nb_:.1f} -> {na_:.1f} deg")

    # ---- not mush: the middle of one dense smooth sphere stays smooth, also after many dabs
    clear_scene()
    obj, per = make_balls([(0, 0, 0)], 64, 32)
    sess = brush.Session(obj)
    st = Settings()
    st.blur_distance = brush.auto_blur_distance(obj)
    plain_w = sess.state.objs[0].plain
    rng = np.random.default_rng(3)
    centre = np.array([0.0, 0.0, 1.0])
    near = np.linalg.norm(sess.vw - centre, axis=1) < 0.4
    def off_smooth():
        dots = np.clip(np.einsum("ij,ij->i", sess.state.vn[near], plain_w[near]), -1, 1)
        return float(np.degrees(np.arccos(dots)).max()), float(np.degrees(np.arccos(dots)).mean())
    first = sess.state.vn.copy()
    for k in range(60):
        p = centre + rng.normal(size=3) * 0.12
        p = p / np.linalg.norm(p)
        sess.dab(tuple(p), 0.35, st)
        if k == 0:
            after1 = sess.state.vn.copy()
    mx, mean = off_smooth()
    info(f"not mush: after 60 dabs mid-sphere, normals vs smooth: max {mx:.2f} deg, mean {mean:.2f} deg")
    check("not mush: dense sphere stays within a few degrees of its smooth normals after 60 dabs", mx < 4.0,
          f"max {mx:.2f} deg")

    # ---- dabs converge: the 20th dab changes less than the first
    a, b = make_two_balls()
    sess = brush.Session([a, b], active=a)
    st = Settings()
    st.blur_distance = brush.auto_blur_distance([a, b])
    seam_pts = sess.vw[cross_seam(sess)[1]]
    sweep = [tuple(p) for p in seam_pts[::3]]
    near_seam = np.linalg.norm(sess.vw[:, None, :] - seam_pts[None, :, :], axis=2).min(axis=1) < 0.5
    changes = []
    for _ in range(20):
        prev = sess.state.vn.copy()
        for p in sweep:
            sess.dab(p, 0.5, st)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", prev, sess.state.vn), -1, 1)))
        changes.append(float(ang[near_seam].mean()))
    info(f"converge: mean normal change per sweep of dabs along the seam, 1st {changes[0]:.3f} deg, 5th {changes[4]:.3f}, 20th {changes[-1]:.3f}")
    check("repeated dabs converge (the 20th sweep moves normals much less than the 1st)", changes[-1] < changes[0] * 0.25,
          f"{changes[0]:.3f} -> {changes[-1]:.3f} deg")

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
    expect = brush.AUTO_FACTOR * float(np.concatenate(lens).mean())
    only_a = brush.AUTO_FACTOR * float(lens[0].mean())
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

    # ---- 1.2.1: only the front side of the mesh gets painted
    def angle_deg(a, b):
        return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))

    # thin slab: two flat sheets 0.05 apart (top faces up, bottom faces down), brush radius 1.2 is way thicker
    clear_scene()
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=24, y_segments=24, size=3.0)
    top_n = len(bm.verts)
    res = bmesh.ops.create_grid(bm, x_segments=24, y_segments=24, size=3.0, matrix=Matrix.Translation((0, 0, -0.05)))
    bmesh.ops.reverse_faces(bm, faces=[f for f in bm.faces if f.verts[0].co.z < -0.01])
    mesh = bpy.data.meshes.new("Slab")
    bm.to_mesh(mesh)
    bm.free()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    slab = bpy.data.objects.new("Slab", mesh)
    bpy.context.scene.collection.objects.link(slab)
    bpy.context.view_layer.objects.active = slab
    slab.select_set(True)
    # start with the top tilted ~20 deg and the bottom plain, so there is something to blur
    nl = len(mesh.loops)
    lv = np.empty(nl, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    vz = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vz)
    is_top = vz.reshape(-1, 3)[lv, 2] > -0.01
    rng = np.random.default_rng(3)
    start = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (nl, 1))
    start[~is_top] = (0, 0, -1)
    start[is_top] += rng.normal(0, 0.18, (int(is_top.sum()), 3)).astype(np.float32)
    start = core_normalize(start)
    brush.prepare_mesh(mesh)
    mesh.normals_split_custom_set(start)
    mesh.update()
    s = Settings()
    s.blur_distance = 0.3
    s.strength = 0.8
    sess = brush.Session(slab)
    ln0 = sess.state.objs[0].ln.copy()
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    err0 = angle_deg(ln0[is_top], np.tile(up, (int(is_top.sum()), 1)))
    for _ in range(3):
        for p in [(0, 0, 0), (0.5, 0.3, 0), (-0.4, 0.2, 0), (0.1, -0.5, 0)]:
            sess.dab(p, 1.2, s)
    ln1 = sess.state.objs[0].ln
    check("thin slab: bottom-side loop normals are bit-identical after dabbing the top",
          np.array_equal(ln1[~is_top], ln0[~is_top]))
    err1 = angle_deg(ln1[is_top], np.tile(up, (int(is_top.sum()), 1)))
    moved_top = np.linalg.norm(ln1[is_top] - ln0[is_top], axis=1) > 1e-6
    info(f"thin slab: top max angle to +up {err0.max():.1f} -> {err1.max():.1f} deg, "
         f"inside the dabbed area (moved loops) {err1[moved_top].max():.1f} deg max, mean {err1[moved_top].mean():.1f}")
    worse = float((err1[moved_top] - err0[moved_top]).max())
    info(f"thin slab: mean angle to +up of the moved loops {err0[moved_top].mean():.1f} -> {err1[moved_top].mean():.1f} deg, "
         f"worst single loop got {worse:+.2f} deg")
    check("thin slab: top normals blur toward +up, nothing flips or goes to garbage",
          moved_top.sum() > 50 and worse < 1.0 and err1[moved_top].mean() < err0[moved_top].mean() * 0.7,
          f"worst {worse:+.2f} deg")
    sess.restore_start()
    for p in [(0, 0, 0), (0.5, 0.3, 0)]:
        sess.dab(p, 1.2, s, "RESTORE")
    check("thin slab: restore leaves the bottom alone too", np.array_equal(sess.state.objs[0].ln[~is_top], ln0[~is_top]))
    clear_scene()

    # sphere, brush bigger than the whole ball
    obj, per = make_balls([(0, 0, 0)], segments=32, rings=16)
    sess = brush.Session(obj)
    st = sess.state
    ln0 = st.objs[0].ln.copy()
    lvs = st.objs[0].loop_vert
    pos = sess.vw[lvs]
    dabs = [(0, 0, 1.0), (0.6, 0, 0.8), (0, 0.6, 0.8)]
    radial_all = core_normalize(pos.astype(np.float32))
    back = np.all([radial_all @ np.array(d, dtype=np.float32) < -0.45 for d in dabs], axis=0)   # behind every dab
    for _ in range(3):
        for p in dabs:
            sess.dab(p, 3.0, s)
    ln1 = st.objs[0].ln
    check("big brush on a sphere: back hemisphere is bit-identical", np.array_equal(ln1[back], ln0[back]),
          f"{int(back.sum())} loops")
    front = pos[:, 2] > 0.3
    radial = core_normalize(pos[front].astype(np.float32))
    e = angle_deg(ln1[front], radial)
    info(f"big brush on a sphere: front max angle to radial {e.max():.2f} deg, mean {e.mean():.2f}")
    check("big brush on a sphere: front stays on the sphere's normals", e.max() < 8, f"{e.max():.2f} deg")
    sess.state.dab((0, 0, 1.0), 3.0, 0.8, 0.3, "RESTORE", hit_normal=(0, 0, 1.0))
    check("big brush restore: back hemisphere bit-identical",
          np.array_equal(st.objs[0].ln[back], ln0[back]))
    clear_scene()

    # mirror dab keeps the facing rule: Mirror X on a thin slab dabbed on the left top leaves both bottoms alone
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=24, y_segments=24, size=3.0)
    bmesh.ops.create_grid(bm, x_segments=24, y_segments=24, size=3.0, matrix=Matrix.Translation((0, 0, -0.05)))
    bmesh.ops.reverse_faces(bm, faces=[f for f in bm.faces if f.verts[0].co.z < -0.01])
    mesh = bpy.data.meshes.new("Slab2")
    bm.to_mesh(mesh)
    bm.free()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    slab = bpy.data.objects.new("Slab2", mesh)
    bpy.context.scene.collection.objects.link(slab)
    slab.rotation_euler = (0.5, 0.2, 0.9)       # rotated, so the mirrored normal needs the local X plane
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = slab
    sess = brush.Session(slab)
    wz = sess.vw
    local_z = np.array([v.co.z for v in mesh.vertices])
    bot_v = local_z < -0.01
    bot_l = bot_v[sess.state.objs[0].loop_vert]
    ln0 = sess.state.objs[0].ln.copy()
    s.mirror_x = True
    hit = slab.matrix_world @ Vector((-0.8, 0.2, 0.0))
    for _ in range(3):
        sess.dab(tuple(hit), 1.2, s)
    s.mirror_x = False
    moved = np.linalg.norm(sess.state.objs[0].ln - ln0, axis=1) > 1e-7
    check("mirror dab on a rotated thin slab: bottoms bit-identical", np.array_equal(sess.state.objs[0].ln[bot_l], ln0[bot_l]))
    clear_scene()

    # negative scale and anisotropic scale + rotation: written normals must match the analytic surface normal
    for label, mat in (("mirrored (scale -1,1,1)", Matrix.Scale(-1, 4, Vector((1, 0, 0)))),
                       ("rotated, scale 2 x 0.5 x 1",
                        Matrix.Rotation(0.8, 4, Vector((1, 2, 3)).normalized())
                        @ Matrix.Diagonal((2.0, 0.5, 1.0, 1.0)))):
        obj, per = make_balls([(0, 0, 0)], segments=32, rings=16)
        obj.matrix_world = mat
        bpy.context.view_layer.update()
        sess = brush.Session(obj)
        m3 = np.array(obj.matrix_world, dtype=np.float64)[:3, :3]
        ctr = sess.vw.mean(axis=0)
        hit_loc = tuple(np.array(obj.matrix_world @ Vector((0, 0, 1.0))))
        for _ in range(3):
            sess.dab(hit_loc, 4.0, s)
        sess.write()
        mesh = obj.data
        ln = read_loop_normals(mesh)
        lvv = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", lvv)
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)[lvv]
        inv_t = np.linalg.inv(m3).T
        got = core_normalize((ln @ inv_t.T).astype(np.float32))            # written local normal -> world
        want = core_normalize((co @ inv_t.T).astype(np.float32))           # analytic: M^-T p for the unit sphere
        if np.linalg.det(m3) < 0:
            pass                                                          # M^-T p is already the outward normal
        e = angle_deg(got, want)
        front = (co[:, 2] > 0.2)
        info(f"{label}: written normals vs analytic surface normal, front max {e[front].max():.2f} deg, "
             f"all max {e.max():.2f} deg")
        check(f"{label}: normals land where the surface normal is", e.max() < 12, f"{e.max():.2f} deg")
        clear_scene()

    # cancellation: a vertex whose plain normal is +x sits between an up-facing and a down-facing sheet,
    # facing can't separate them (both ~90 deg), the sum cancels and the vertex has to stay put
    from EmilsNormalBrush import core as ncore
    verts = np.array([[-1, -1, 0.01], [1, -1, 0.01], [0, 1, 0.01],
                      [-1, -1, -0.01], [1, -1, -0.01], [0, 1, -0.01],
                      [0, 0, 0]], dtype=np.float32)
    tris = np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64)               # up sheet, down sheet
    plain = np.array([[0, 0, 1]] * 3 + [[0, 0, -1]] * 3 + [[1, 0, 0]], dtype=np.float32)
    loop_vert = np.arange(7)
    ln = plain.copy()
    on = ncore.ObjectNormals(verts, loop_vert, ln, plain)
    st = ncore.NormalBrushState([on], tris=tris)
    st.dab((0, 0, 0), 2.0, 1.0, 0.5)
    check("cancelling neighbourhood: the vertex isn't moved", np.array_equal(on.ln[6], ln[6]),
          str(on.ln[6]))
    check("cancelling neighbourhood: the others still blur normally", on.touched)
    clear_scene()

    # ---- 1.3.0: paint what you SEE (shape keys, Mirror modifier, stacked layers)
    from EmilsNormalBrush import core as ncore

    def evaluated(obj, modifiers=False):
        """(local positions, local loop normals) of the depsgraph-evaluated mesh. Modifiers are switched off
        for the evaluation unless asked for, and switched back on afterwards."""
        state = [(m, m.show_viewport) for m in obj.modifiers]
        if not modifiers:
            for m, _ in state:
                m.show_viewport = False
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        me.calc_normals_split()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        ln = read_loop_normals(me)
        lv = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", lv)
        bpy.data.meshes.remove(me)
        for m, v in state:
            m.show_viewport = v
        bpy.context.view_layer.update()
        return co.reshape(-1, 3), ln, lv

    def make_shape_sphere(with_mirror=False):
        clear_scene()
        obj, per = make_balls([(0, 0, 0)], segments=32, rings=16)
        obj.matrix_world = Matrix.Translation((0.3, -0.2, 0.1)) @ Matrix.Rotation(0.4, 4, "Z") @ Matrix.Scale(1.4, 4)
        mesh = obj.data
        co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
        # custom normals to start with, not just the smooth ones
        wob = co + 0.25 * np.stack([np.sin(3 * co[:, 0]), np.cos(2 * co[:, 1]), np.sin(4 * co[:, 2])], axis=1)
        brush.prepare_mesh(mesh)
        lvv = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", lvv)
        mesh.normals_split_custom_set(core_normalize(wob.astype(np.float32))[lvv])
        obj.shape_key_add(name="Basis")
        k1 = obj.shape_key_add(name="Twist")
        ang = np.radians(40.0) * np.clip(co[:, 2] / 0.6, 0.0, 1.0)          # tip of the sphere tilts ~40 deg about X
        cy, sy = np.cos(ang), np.sin(ang)
        moved = co.copy()
        moved[:, 1] = co[:, 1] * cy - co[:, 2] * sy
        moved[:, 2] = co[:, 1] * sy + co[:, 2] * cy
        for i, p in enumerate(moved):
            k1.data[i].co = p
        k1.value = 1.0
        k2 = obj.shape_key_add(name="Bulge")
        for i, p in enumerate(co):
            k2.data[i].co = p * (1.0 + 0.2 * (p[0] > 0))
        k2.value = 0.5
        vg = obj.vertex_groups.new(name="Right")
        for i, p in enumerate(co):
            if p[0] > 0:
                vg.add([i], 0.8, "REPLACE")
        k2.vertex_group = "Right"
        obj.active_shape_key_index = 1
        if with_mirror:
            m = obj.modifiers.new("Mir", "MIRROR")
            m.use_axis = (True, False, False)
        bpy.context.view_layer.update()
        return obj

    # displayed positions against Blender's own evaluation (modifiers off), also with a Mirror modifier on
    obj = make_shape_sphere(with_mirror=True)
    ev_co, _, _ = evaluated(obj)
    mine = brush.displayed_local(obj)
    err = float(np.abs(mine - ev_co).max())
    base = brush.basis_local(obj)
    check("shape keys: displayed positions match Blender's evaluation (value, relative key, vertex group)",
          err < 1e-5 and np.abs(base - ev_co).max() > 0.05, f"max err {err:.2e}, shape key moved things up to "
          f"{np.abs(base - ev_co).max():.2f}")
    obj.show_only_shape_key = True
    obj.active_shape_key_index = 1
    ev_co, _, _ = evaluated(obj)
    err = float(np.abs(brush.displayed_local(obj) - ev_co).max())
    check("shape keys: 'show only active shape key' matches too", err < 1e-5, f"max err {err:.2e}")
    obj.show_only_shape_key = False
    obj.data.shape_keys.key_blocks["Twist"].mute = True
    ev_co, _, _ = evaluated(obj)
    err = float(np.abs(brush.displayed_local(obj) - ev_co).max())
    check("shape keys: a muted key is skipped", err < 1e-5, f"max err {err:.2e}")
    obj.data.shape_keys.key_blocks["Twist"].mute = False

    # custom normals follow the deformation: displayed state vs Blender's evaluated normals
    obj = make_shape_sphere()
    ev_co, ev_ln, ev_lv = evaluated(obj)
    sess = brush.Session(obj)
    sl = sess.state.objs[0]
    mw = np.array(obj.matrix_world, dtype=np.float64)
    check("shape keys: Session positions are the displayed ones",
          np.abs(sess.vw - (ev_co.astype(np.float64) @ mw[:3, :3].T + mw[:3, 3])).max() < 1e-4)
    e0 = angle_deg(sl.ln, ev_ln)
    base_ln = read_loop_normals(obj.data)
    e_plain = angle_deg(base_ln, ev_ln)
    info(f"shape key normals, nothing painted: stored->displayed rotation vs Blender's evaluation: mean {e0.mean():.2f} "
         f"max {e0.max():.2f} deg (not rotating at all would be mean {e_plain.mean():.1f} max {e_plain.max():.1f} deg)")
    worst = int(np.argmax(e0))
    info(f"shape key normals, nothing painted: 99th percentile {np.percentile(e0, 99):.2f} deg, the worst loop sits at "
         f"local z {ev_co[ev_lv[worst]][2]:.2f} (original {ev_co[ev_lv[worst]].tolist()})")
    check("shape keys: the rotated stored normals match what Blender displays",
          e0.mean() < 1.0 and np.percentile(e0, 99) < 4.0 and e0.max() < 8.0, f"mean {e0.mean():.2f}, max {e0.max():.2f} deg")
    s = Settings()
    s.blur_distance = 0.3
    s.strength = 0.8
    top = brush.world_positions(obj)
    tip = tuple(top[np.argmax(top[:, 2])])
    for _ in range(3):
        for off in [(0, 0, 0), (0.3, 0.1, -0.2), (-0.3, -0.2, -0.3), (0, 0.4, -0.4)]:
            sess.dab(tuple(np.array(tip) + off), 0.9, s)
    moved = np.linalg.norm(sl.ln - sl.start_ln, axis=1) > 1e-4
    sess.write()
    _, ev_ln2, _ = evaluated(obj)
    e1 = angle_deg(sl.ln, ev_ln2)
    info(f"shape key normals after painting ({int(moved.sum())} loops moved): written -> evaluated vs intended: "
         f"mean {e1.mean():.2f} max {e1.max():.2f} deg, on the painted loops mean {e1[moved].mean():.2f} "
         f"max {e1[moved].max():.2f}")
    check("shape keys: painted normals show up where intended once Blender evaluates the shape key",
          moved.sum() > 100 and e1[moved].max() < 3.0 and e1.mean() < 1.0 and e1.max() < 8.0,
          f"painted max {e1[moved].max():.2f} deg, overall max {e1.max():.2f}")
    # the numpy rotation itself is exactly invertible, also for a == b and a == -b
    a = core_normalize(np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0.3, 0.2, 0.9]], dtype=np.float32)).astype(np.float64)
    b = core_normalize(np.array([[0, 0, 1], [0, 0, -1], [1, 1e-9, 0], [-0.4, 0.5, 0.2]], dtype=np.float32)).astype(np.float64)
    v = core_normalize(np.array([[1, 2, 3], [1, 2, 3], [0, 1, 1], [3, 1, -2]], dtype=np.float32)).astype(np.float64)
    fwd = ncore.rotate_arc(v, a, b)
    back = ncore.rotate_arc(fwd, b, a)
    check("rotate_arc: a->b maps a onto b, b->a undoes it, a == b and a == -b are fine",
          np.abs(ncore.rotate_arc(a, a, b) - b).max() < 1e-5 and np.abs(back - v).max() < 1e-5
          and np.allclose(np.linalg.norm(fwd, axis=1), 1.0), f"{np.abs(back - v).max():.1e}")
    clear_scene()

    # what does the 3.6 Mirror modifier do with custom normals?
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=4, y_segments=4, size=1.0, matrix=Matrix.Translation((1.5, 0, 0)))
    mesh = bpy.data.meshes.new("MirrorN")
    bm.to_mesh(mesh)
    bm.free()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mobj = bpy.data.objects.new("MirrorN", mesh)
    bpy.context.scene.collection.objects.link(mobj)
    brush.prepare_mesh(mesh)
    mesh.normals_split_custom_set(np.tile(core_normalize(np.array([[0.5, 0.2, 0.84]], dtype=np.float32)),
                                          (len(mesh.loops), 1)))
    mm = mobj.modifiers.new("Mir", "MIRROR")
    mm.use_axis = (True, False, False)
    mm.use_mirror_merge = False
    ev_co, ev_ln, ev_lv = evaluated(mobj, modifiers=True)
    left = ev_co[ev_lv][:, 0] < 0
    want_mirrored = core_normalize(np.array([[-0.5, 0.2, 0.84]], dtype=np.float32))[0]
    want_same = core_normalize(np.array([[0.5, 0.2, 0.84]], dtype=np.float32))[0]
    got = ev_ln[left].mean(axis=0)
    got /= np.linalg.norm(got)
    to_mir = float(np.degrees(np.arccos(np.clip(got @ want_mirrored, -1, 1))))
    to_same = float(np.degrees(np.arccos(np.clip(got @ want_same, -1, 1))))
    info(f"Mirror modifier and custom normals: mirrored loops (x<0) have normal {np.round(got, 3).tolist()}, "
         f"{to_mir:.2f} deg from the mirrored normal, {to_same:.1f} deg from the unmirrored copy "
         f"-> Blender {'DOES' if to_mir < 2 else 'does NOT'} mirror custom normals")
    clear_scene()

    # a click on the Mirror modifier's copy lands on the real half, and the copy's faces take part in the blur
    def half_sphere(mirror):
        clear_scene()
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, 0), plane_no=(1, 0, 0), clear_inner=True, dist=1e-5)
        mesh = bpy.data.meshes.new("Half")
        bm.to_mesh(mesh)
        bm.free()
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        ob = bpy.data.objects.new("Half", mesh)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        if mirror:
            m = ob.modifiers.new("Mir", "MIRROR")
            m.use_axis = (True, False, False)
        # start with scrambled custom normals so the dabs have something to fix
        brush.prepare_mesh(mesh)
        lvv = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", lvv)
        co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
        r = np.random.default_rng(5).normal(0, 0.35, co.shape).astype(np.float32)
        mesh.normals_split_custom_set(core_normalize(co + r)[lvv])
        bpy.context.view_layer.update()
        return ob

    ob = half_sphere(True)
    sess = brush.Session(ob)
    check("Mirror modifier: the session knows one mirrored copy", len(sess.ghosts) == 1, str(len(sess.ghosts)))
    p = sess.pick((-0.6, 0.0, 5.0), (0.0, 0.0, -1.0))
    ok_pick = (p is not None and p.loc[0] > 0.3 and p.shown[0] < -0.3 and abs(p.loc[0] + p.shown[0]) < 1e-4
               and p.normal[0] > 0 and p.normal[2] > 0 and p.view[0][0] > 0)
    check("Mirror modifier: a click on the copy maps back to the real half (point, normal, view)", ok_pick,
          f"shown x {p.shown[0]:.2f} -> real x {p.loc[0]:.2f}" if p else "miss")
    p_real = sess.pick((0.6, 0.0, 5.0), (0.0, 0.0, -1.0))
    check("Mirror modifier: a click on the real half is left alone", p_real is not None and np.allclose(p_real.loc, p_real.shown))
    s = Settings()
    s.blur_distance = 0.3
    s.strength = 0.8
    sl = sess.state.objs[0]
    before_ln = sl.ln.copy()
    for _ in range(3):
        sess.dab(p.loc, 0.5, s, "BLUR", p.normal, p.view)
    mv = np.nonzero(np.linalg.norm(sl.ln - before_ln, axis=1) > 1e-5)[0]
    mx = sess.vw[sl.loop_vert[mv]][:, 0]
    check("Mirror modifier: painting the copy changes real vertices on the real side only", len(mv) > 20 and (mx > 0).all(),
          f"{len(mv)} loops, x {mx.min():.2f}..{mx.max():.2f}")
    # blur target includes the mirror image: at the mirror plane the painted normals come out symmetric (nx ~ 0)
    def plane_error(with_mirror):
        ob2 = half_sphere(with_mirror)
        sess2 = brush.Session(ob2)
        sl2 = sess2.state.objs[0]
        on_plane = np.abs(sess2.vw[sl2.loop_vert][:, 0]) < 1e-4
        for _ in range(5):
            for pt in [(0.0, 0.0, 1.0), (0.0, 0.7, 0.7), (0.0, -0.7, 0.7), (0.0, 0.9, 0.0), (0.0, -0.9, 0.0)]:
                sess2.dab(pt, 0.5, s, "BLUR", (1.0, 0.0, 0.0))
        return float(np.degrees(np.arcsin(np.clip(np.abs(sl2.ln[on_plane][:, 0]), 0, 1))).mean())
    e_with, e_without = plane_error(True), plane_error(False)
    info(f"Mirror modifier blur at the mirror plane: mean angle off the plane's symmetry {e_without:.1f} deg without, "
         f"{e_with:.1f} deg with the copy's faces")
    check("Mirror modifier: faces of the copy are blurred against (plane normals end up more symmetric)",
          e_with < e_without, f"{e_without:.1f} -> {e_with:.1f}")
    clear_scene()

    # feathers: three stacked layers 0.02 apart under a much bigger brush, painted from above
    bm = bmesh.new()
    for z in (0.0, -0.02, -0.04):
        bmesh.ops.create_grid(bm, x_segments=20, y_segments=20, size=3.0, matrix=Matrix.Translation((0, 0, z)))
    mesh = bpy.data.meshes.new("Feathers")
    bm.to_mesh(mesh)
    bm.free()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    fe = bpy.data.objects.new("Feathers", mesh)
    bpy.context.scene.collection.objects.link(fe)
    bpy.context.view_layer.objects.active = fe
    lvv = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lvv)
    co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    layer_v = np.round(-co[:, 2] / 0.02).astype(int)
    brush.prepare_mesh(mesh)
    wob = np.stack([0.5 * np.sin(4 * co[:, 0]), 0.5 * np.cos(3 * co[:, 1]), np.ones(len(co))], axis=1).astype(np.float32)
    mesh.normals_split_custom_set(core_normalize(wob)[lvv])
    bpy.context.view_layer.update()
    s = Settings()
    s.blur_distance = 0.3
    s.strength = 0.8
    for label, persp, view in (("perspective", True, ((0.0, 0.0, 6.0), (0.0, 0.0, -1.0))),
                               ("ortho", False, ((0.0, 0.0, 6.0), (0.0, 0.0, -1.0)))):
        sess = brush.Session(fe)
        sl = sess.state.objs[0]
        l_of_loop = layer_v[sl.loop_vert]
        ln0 = sl.ln.copy()
        for pt in [(0, 0, 0), (0.5, 0.4, 0), (-0.6, 0.2, 0)]:
            sess.dab(pt, 2.0, s, "BLUR", (0, 0, 1.0), (view[0], view[1], persp))
        top_moved = int((np.linalg.norm(sl.ln - ln0, axis=1)[l_of_loop == 0] > 1e-5).sum())
        check(f"feathers ({label} view): lower two layers are bit-identical, top layer gets painted",
              np.array_equal(sl.ln[l_of_loop > 0], ln0[l_of_loop > 0]) and top_moved > 100, f"{top_moved} top loops moved")
        sess.restore_start()
    sess = brush.Session(fe)
    sl = sess.state.objs[0]
    ln0 = sl.ln.copy()
    sess.dab((0, 0, 0), 2.0, s, "BLUR", (0, 0, 1.0))
    check("feathers: without a view (no occlusion check) the lower layers do get painted too",
          not np.array_equal(sl.ln[l_of_loop > 0], ln0[l_of_loop > 0]))
    dv = sess.visible_fn(((0.0, 0.0, 6.0), (0.0, 0.0, -1.0), True))
    pts = sess.vw[:600].copy()
    t0 = time.perf_counter()
    for _ in range(5):
        dv(pts, 2.0)
    info(f"occlusion check: {(time.perf_counter() - t0) / 5 * 1000:.1f} ms for 600 vertices")
    clear_scene()

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
