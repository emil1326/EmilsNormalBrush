"""
Emil's Normal Brush: the tool, the stroke operator, the radius/strength adjuster and the cursor.
"""

import math
import time

import bpy
import gpu
import numpy as np
from bpy.props import EnumProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import core

WRITE_INTERVAL = 1.0 / 30.0   # max writes to the mesh per second
AUTO_FACTOR = 1.5             # Auto Blur Distance = this x mean edge length
OCCLUDE_TOL = 0.001          # a face this much (x brush radius) in front of a vertex hides it; layers can sit 0.02 apart under a big brush
SPACING = 0.15                # dab spacing, fraction of the brush radius (in pixels)

_hover_cache = {}             # per object name, see hover_data
_cache_version = 0            # bumped whenever normals change outside the hover cache's sight
_last_cursor = {"xy": None, "calls": 0}   # what draw_cursor last received (debug / GUI test)


def bump_cache():
    global _cache_version
    _cache_version += 1


# ----------------------------------------------------------------- stroke session

def prepare_mesh(mesh):
    """Custom normals need auto smooth in 3.6. Turn it on with a 180 degree angle only if it was off."""
    if not mesh.use_auto_smooth:
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.pi


def _key_co(kb, n):
    co = np.empty(n * 3, dtype=np.float32)
    kb.data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def basis_local(obj):
    """Local vertex positions of the basis shape (what mesh.vertices holds), float64."""
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def _group_weights(obj, name):
    """Per vertex weight of vertex group `name` (0 where the vertex isn't in it)."""
    mesh = obj.data
    vg = obj.vertex_groups.get(name)
    w = np.zeros(len(mesh.vertices), dtype=np.float64)
    if vg is None:
        return np.ones(len(mesh.vertices), dtype=np.float64)     # Blender ignores a missing group
    gi = vg.index
    for i, v in enumerate(mesh.vertices):
        for g in v.groups:
            if g.group == gi:
                w[i] = g.weight
                break
    return w


def displayed_local(obj):
    """Local vertex positions as Blender shows them before any modifier: the shape key mix.
    Each non-muted key adds value * group weight * (key - its relative key); with 'show only active shape key'
    just that key shows. Without shape keys this is the plain vertex positions."""
    mesh = obj.data
    n = len(mesh.vertices)
    sk = mesh.shape_keys
    if sk is None or len(sk.key_blocks) == 0:
        return basis_local(obj)
    ref = sk.reference_key
    if obj.show_only_shape_key and obj.active_shape_key is not None:
        return _key_co(obj.active_shape_key, n)
    co = _key_co(ref, n)
    if not sk.use_relative:
        return co
    for kb in sk.key_blocks:
        if kb.as_pointer() == ref.as_pointer() or kb.mute or kb.value == 0.0:
            continue
        delta = _key_co(kb, n) - _key_co(kb.relative_key, n)
        w = kb.value
        if kb.vertex_group:
            co += (w * _group_weights(obj, kb.vertex_group))[:, None] * delta
        else:
            co += w * delta
    return co


def world_positions(obj, local=None):
    """World positions of the displayed (shape key mixed, modifier free) vertices."""
    co = displayed_local(obj) if local is None else local
    mw = np.array(obj.matrix_world, dtype=np.float64)
    return (co @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)


def mesh_edges(mesh):
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", ev)
    return ev.reshape(-1, 2)


def poly_arrays(mesh):
    start = np.empty(len(mesh.polygons), dtype=np.int64)
    total = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", start)
    mesh.polygons.foreach_get("loop_total", total)
    return start, total


# Modifiers that change the topology: a Mirror after one of these isn't a plain mirror of the base mesh any more.
TOPOLOGY_MODIFIERS = {"SUBSURF", "ARRAY", "BEVEL", "BOOLEAN", "BUILD", "DECIMATE", "EDGE_SPLIT", "MASK", "MULTIRES",
                      "REMESH", "SCREW", "SOLIDIFY", "SKIN", "TRIANGULATE", "WIREFRAME", "WELD", "NODES",
                      "VOLUME_TO_MESH", "MESH_TO_VOLUME", "EXPLODE", "OCEAN", "PARTICLE_SYSTEM"}


def mirror_matrices(obj):
    """World-space 4x4 reflections, one per extra copy the Mirror modifiers (before the first topology changing
    modifier) add. Several axes on one modifier give the X, Y, XY... copies, a second Mirror mirrors the first
    one's copies as well. Bisect and flip are ignored."""
    copies = [np.identity(4)]
    for mod in obj.modifiers:
        if mod.type in TOPOLOGY_MODIFIERS:
            break
        if mod.type != "MIRROR" or not mod.show_viewport:
            continue
        axes = [i for i in range(3) if mod.use_axis[i]]
        if not axes:
            continue
        ref = np.array((mod.mirror_object or obj).matrix_world, dtype=np.float64)
        ref_inv = np.linalg.inv(ref)
        subsets = [[a for k, a in enumerate(axes) if mask >> k & 1] for mask in range(1, 1 << len(axes))]
        added = []
        for sub in subsets:
            s = np.identity(4)
            for a in sub:
                s[a, a] = -1.0
            w = ref @ s @ ref_inv
            added += [w @ c for c in copies]
        copies += added
        if len(copies) > 9:
            copies = copies[:9]
            break
    return copies[1:]


def unique_by_data(objs):
    """Keep the first object of each mesh datablock (linked duplicates share one)."""
    seen, out = set(), []
    for ob in objs:
        if ob.data.as_pointer() not in seen:
            seen.add(ob.data.as_pointer())
            out.append(ob)
    return out


def selected_meshes(context, dedupe=True):
    """Active mesh first (if it is one), then every other selected mesh object."""
    objs = []
    active = context.active_object
    if active is not None and active.type == "MESH":
        objs.append(active)
    for ob in context.selected_objects:
        if ob.type == "MESH" and ob not in objs:
            objs.append(ob)
    return unique_by_data(objs) if dedupe else objs


def auto_blur_distance(objs):
    """AUTO_FACTOR x the mean edge length (world space) over one object or a list of them."""
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    total, count = 0.0, 0
    for ob in unique_by_data(objs):
        ev = mesh_edges(ob.data)
        if len(ev) == 0:
            continue
        wp = world_positions(ob)
        total += float(np.linalg.norm(wp[ev[:, 0]] - wp[ev[:, 1]], axis=1).sum())
        count += len(ev)
    return AUTO_FACTOR * total / count if count else 0.0


class Pick:
    """What a click hit, in the space of the REAL mesh: location, face normal, and the view the user sees it from.
    `shown` is where the user sees it (differs from `loc` on a Mirror modifier's copy)."""

    def __init__(self, loc, normal, shown, origin, direction, perspective):
        self.loc, self.normal, self.shown = loc, normal, shown
        self.view = (origin, direction, perspective)


class Session:
    """Everything one stroke needs: the numpy state over all objects, one BVH to raycast, the write-back."""

    def __init__(self, objs, active=None):
        if isinstance(objs, bpy.types.Object):
            objs = [objs]
        self.objects = [ob for ob in unique_by_data(list(objs)) if len(ob.data.loops) > 0]
        if not self.objects:
            raise ValueError("no mesh with faces to paint")
        slices = []
        for obj in self.objects:
            mesh = obj.data
            prepare_mesh(mesh)
            mesh.calc_normals_split()
            nloops, nverts = len(mesh.loops), len(mesh.vertices)
            ln = np.empty(nloops * 3, dtype=np.float32)
            mesh.loops.foreach_get("normal", ln)
            lv = np.empty(nloops, dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", lv)
            pn = np.empty(nverts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("normal", pn)
            disp = displayed_local(obj)
            base = basis_local(obj)
            rot = None
            if np.abs(disp - base).max(initial=0.0) > 1e-7:     # a shape key moved things: normals follow the surface
                ps, pt = poly_arrays(mesh)
                nb_, nd_ = core.vertex_normals(base, lv, ps, pt), core.vertex_normals(disp, lv, ps, pt)
                ref_v = core.ref_neighbors(lv, ps, pt, nverts)
                rot = (core.vertex_frames(base, nb_, ref_v), core.vertex_frames(disp, nd_, ref_v))
                pn = nd_
            slices.append(core.ObjectNormals(world_positions(obj, disp), lv, ln.reshape(-1, 3), pn.reshape(-1, 3),
                                             obj.matrix_world, rot=rot))
        ref = self.objects.index(active) if active in self.objects else 0
        offsets = np.concatenate([[0], np.cumsum([s.nverts for s in slices])]).astype(np.int64)
        real_tris = self._triangles(offsets)
        self.n_real_tris = len(real_tris)
        vw_real = np.concatenate([s.vw for s in slices])
        # the Mirror modifier's copies: static ghost triangles, they are raycast and blurred against
        ghost_vw, ghost_tris, self.ghosts = [], [], []
        gv_off, t_off = len(vw_real), len(real_tris)
        for obj, sl, off, nxt in zip(self.objects, slices, offsets[:-1], offsets[1:]):
            own = real_tris[(real_tris >= off).all(axis=1) & (real_tris < nxt).all(axis=1)]
            for w in mirror_matrices(obj):
                gp = (sl.vw.astype(np.float64) @ w[:3, :3].T + w[:3, 3]).astype(np.float32)
                t = own - int(off) + gv_off
                if np.linalg.det(w[:3, :3]) < 0:
                    t = t[:, [0, 2, 1]]
                ghost_vw.append(gp)
                ghost_tris.append(t)
                self.ghosts.append((w, t_off, len(t)))
                gv_off += len(gp)
                t_off += len(t)
        self.tris = real_tris if not ghost_tris else np.concatenate([real_tris] + ghost_tris)
        gvw = np.concatenate(ghost_vw) if ghost_vw else None
        self.state = core.NormalBrushState(slices, mirror_ref=ref, tris=self.tris, ghost_vw=gvw)
        self.vw = self.state.vw
        allw = self.vw if gvw is None else np.concatenate([self.vw, gvw])
        self.bvh = BVHTree.FromPolygons(allw.tolist(), self.tris.tolist())

    @property
    def dirty(self):
        return any(o.changed for o in self.state.objs)

    def _triangles(self, offsets):
        """All triangles of all objects as vertex indices into the concatenated positions."""
        tris = []
        for obj, off in zip(self.objects, offsets):
            mesh = obj.data
            mesh.calc_loop_triangles()
            t = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", t)
            t = t.reshape(-1, 3).astype(np.int64) + int(off)
            if np.linalg.det(np.array(obj.matrix_world, dtype=np.float64)[:3, :3]) < 0:
                t = t[:, [0, 2, 1]]           # mirrored object (negative scale): keep the world winding outward
            tris.append(t)
        return np.concatenate(tris)

    def raycast(self, origin, direction):
        pick = self.pick(origin, direction)
        return None if pick is None else pick.loc

    def pick(self, origin, direction, perspective=True):
        """Raycast the view ray. A hit on a Mirror modifier's copy is mapped back to the real mesh (point, normal
        and the view ray itself), so the dab lands where the vertices really are."""
        loc, normal, index, _dist = self.bvh.ray_cast(Vector(origin), Vector(direction))
        if loc is None:
            return None
        o, d = np.array(origin, dtype=np.float64), np.array(direction, dtype=np.float64)
        shown = np.array(loc, dtype=np.float64)
        p, n = shown.copy(), np.array(normal, dtype=np.float64)
        if index >= self.n_real_tris:
            for w, start, count in self.ghosts:
                if start <= index < start + count:
                    winv = np.linalg.inv(w)
                    p = (winv @ np.append(shown, 1.0))[:3]
                    n = w[:3, :3].T @ n
                    n /= max(np.linalg.norm(n), 1e-12)
                    o = (winv @ np.append(o, 1.0))[:3]
                    d = winv[:3, :3] @ d
                    d /= max(np.linalg.norm(d), 1e-12)
                    break
        return Pick(tuple(p), tuple(n), tuple(shown), o, d, perspective)

    def visible_fn(self, view):
        """Returns visible(points, radius) -> bool mask: False for points a face clearly in front of them hides
        from the viewer (a ray from the eye to the point hits something more than OCCLUDE_TOL x radius sooner)."""
        origin, direction, perspective = view
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        ray_cast = self.bvh.ray_cast

        def visible(points, radius):
            pts = np.asarray(points, dtype=np.float64)
            tol = max(OCCLUDE_TOL * radius, 1e-6)
            if perspective:
                vec = pts - origin
                dist = np.linalg.norm(vec, axis=1)
                dirs = vec / np.maximum(dist, 1e-12)[:, None]
                starts = np.broadcast_to(origin, pts.shape)
            else:                                           # ortho: parallel rays from the plane through `origin`
                dist = (pts - origin) @ direction
                dirs = np.broadcast_to(direction, pts.shape)
                starts = pts - dirs * dist[:, None]
            ok = np.ones(len(pts), dtype=bool)
            s_l, d_l, m_l = starts.tolist(), dirs.tolist(), (dist + tol).tolist()
            lim = (dist - tol).tolist()
            for i in range(len(pts)):
                loc = ray_cast(s_l[i], d_l[i], m_l[i])[0]
                if loc is not None:
                    s = s_l[i]
                    if ((loc[0] - s[0]) ** 2 + (loc[1] - s[1]) ** 2 + (loc[2] - s[2]) ** 2) ** 0.5 < lim[i]:
                        ok[i] = False
            return ok

        return visible

    def dab(self, hit_world, world_radius, settings, mode="BLUR", hit_normal=None, view=None):
        """view = (origin, direction, perspective) of the viewer. No view, or Only Visible off, skips the occlusion check."""
        if hit_normal is None:                       # no ray at hand (tests): use the face closest to the hit
            _loc, hit_normal, _i, _d = self.bvh.find_nearest(Vector(hit_world))
        occlude = view is not None and getattr(settings, "only_visible", True)
        visible = self.visible_fn(view) if occlude else None
        return self.state.dab(hit_world, world_radius, settings.strength, settings.blur_distance,
                              mode, settings.mirror_x, hit_normal, visible)

    def write(self):
        """Write every object whose normals changed since the last write."""
        wrote = False
        for obj, sl in zip(self.objects, self.state.objs):
            if not sl.changed:
                continue
            obj.data.normals_split_custom_set(sl.base_normals())
            obj.data.update()
            sl.changed = False
            wrote = True
        return wrote

    def restore_start(self):
        """Esc: put every object that was touched back to its start normals."""
        for obj, sl in zip(self.objects, self.state.objs):
            if not sl.touched:
                continue
            sl.reset()
            obj.data.normals_split_custom_set(sl.base_normals())
            obj.data.update()
            sl.changed = sl.touched = False
        self.state.reset()


def view_ray(context, coord):
    region, rv3d = context.region, context.region_data
    return (view3d_utils.region_2d_to_origin_3d(region, rv3d, coord),
            view3d_utils.region_2d_to_vector_3d(region, rv3d, coord))


def world_radius_at(context, coord, hit, radius_px):
    region, rv3d = context.region, context.region_data
    a = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, hit)
    b = view3d_utils.region_2d_to_location_3d(region, rv3d, (coord[0] + radius_px, coord[1]), hit)
    return (b - a).length


# ----------------------------------------------------------------- operators

class EMILSNB_OT_stroke(bpy.types.Operator):
    bl_idname = "emils.normal_brush_stroke"
    bl_label = "Normal Brush Stroke"
    bl_description = "Blur (or with Ctrl, restore) the custom normals under the brush"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(name="Mode", items=[("BLUR", "Blur", ""), ("RESTORE", "Restore", "")], default="BLUR")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D" and context.region_data is not None

    def invoke(self, context, event):
        objs = selected_meshes(context)
        if not objs or context.mode != "OBJECT":
            self.report({"WARNING"}, "Select a mesh and stay in Object Mode")
            return {"CANCELLED"}
        if not any(len(ob.data.loops) for ob in objs):
            self.report({"WARNING"}, "The mesh has no faces")
            return {"CANCELLED"}
        self.session = Session(objs, active=context.active_object)
        self.last_write = 0.0
        self.last_xy = None
        self._dab_at(context, (event.mouse_region_x, event.mouse_region_y))
        self.session.write()
        self.last_write = time.perf_counter()
        wm = context.window_manager
        self.timer = wm.event_timer_add(WRITE_INTERVAL, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _dab_at(self, context, xy):
        settings = context.scene.emils_normal_brush
        origin, direction = view_ray(context, xy)
        pick = self.session.pick(origin, direction, context.region_data.is_perspective)
        self.last_xy = xy
        if pick is None:
            return False
        r = world_radius_at(context, xy, Vector(pick.shown), settings.radius)
        return self.session.dab(pick.loc, r, settings, self.mode, pick.normal, pick.view)

    def _finish(self, context):
        context.window_manager.event_timer_remove(self.timer)
        bump_cache()

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.session.restore_start()
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self.session.write()
            self._finish(context)
            return {"FINISHED"}
        if event.type == "MOUSEMOVE":
            settings = context.scene.emils_normal_brush
            xy = (event.mouse_region_x, event.mouse_region_y)
            gap = math.hypot(xy[0] - self.last_xy[0], xy[1] - self.last_xy[1])
            step = max(2.0, SPACING * settings.radius)
            if gap >= step:
                steps = min(int(gap // step), 8)
                x0, y0 = self.last_xy
                for i in range(1, steps + 1):
                    t = i * step / gap
                    self._dab_at(context, (x0 + (xy[0] - x0) * t, y0 + (xy[1] - y0) * t))
        if self.session.dirty and time.perf_counter() - self.last_write >= WRITE_INTERVAL:
            self.session.write()
            self.last_write = time.perf_counter()
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}


class EMILSNB_OT_adjust(bpy.types.Operator):
    """Change the brush radius or strength by moving the mouse sideways"""
    bl_idname = "emils.normal_brush_adjust"
    bl_label = "Adjust Normal Brush"
    bl_options = {"INTERNAL"}

    target: EnumProperty(name="Target", items=[("RADIUS", "Radius", ""), ("STRENGTH", "Strength", "")],
                         default="RADIUS")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        self.settings = context.scene.emils_normal_brush
        self.start_x = event.mouse_x
        self.center = (event.mouse_region_x, event.mouse_region_y)
        self.initial = getattr(self.settings, self.target.lower())
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {"RUNNING_MODAL"}

    def _draw(self):
        s = self.settings
        draw_circle(self.center[0], self.center[1], s.radius, (1.0, 1.0, 1.0, 0.9))
        draw_circle(self.center[0], self.center[1], s.radius * s.strength, (1.0, 1.0, 1.0, 0.35))

    def _header(self, context):
        s = self.settings
        context.area.header_text_set(f"Normal Brush  radius {s.radius}px  strength {s.strength:.2f}"
                                     "   (click or Enter to confirm, Esc to cancel)")

    def _end(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            dx = event.mouse_x - self.start_x
            if self.target == "RADIUS":
                self.settings.radius = max(5, min(1000, int(self.initial + dx)))
            else:
                self.settings.strength = max(0.0, min(1.0, self.initial + dx / 400.0))
            self._header(context)
            context.area.tag_redraw()
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._end(context)
            return {"FINISHED"}
        elif event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            setattr(self.settings, self.target.lower(), self.initial)
            self._end(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


class EMILSNB_OT_auto_distance(bpy.types.Operator):
    bl_idname = "emils.normal_brush_auto_distance"
    bl_label = "Auto Blur Distance"
    bl_description = "Set the blur distance to about 1.5 times the mean edge length of the selected meshes (world space)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(selected_meshes(context))

    def execute(self, context):
        value = auto_blur_distance(selected_meshes(context))
        if value <= 0:
            self.report({"WARNING"}, "No edges to measure")
            return {"CANCELLED"}
        context.scene.emils_normal_brush.blur_distance = value
        return {"FINISHED"}


class EMILSNB_OT_clear(bpy.types.Operator):
    bl_idname = "emils.normal_brush_clear"
    bl_label = "Clear Custom Normals"
    bl_description = "Remove the custom split normals of every selected mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(selected_meshes(context))

    def execute(self, context):
        done = False
        for ob in selected_meshes(context):
            with context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
                result = bpy.ops.mesh.customdata_custom_splitnormals_clear()
            done |= "FINISHED" in result
        bump_cache()
        return {"FINISHED"} if done else {"CANCELLED"}


# ----------------------------------------------------------------- cursor drawing

_shader = None


def _flat_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _shader


def draw_circle(x, y, radius, color, segments=64):
    pts = [(x + math.cos(2 * math.pi * i / segments) * radius,
            y + math.sin(2 * math.pi * i / segments) * radius) for i in range(segments + 1)]
    shader = _flat_shader()
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": pts})
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.5)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def _shape_signature(obj):
    sk = obj.data.shape_keys
    if sk is None:
        return None
    return (obj.show_only_shape_key, obj.active_shape_key_index,
            tuple((kb.value, kb.mute) for kb in sk.key_blocks))


def hover_data(obj):
    """World positions and unit vertex normals (world space) of obj, cached until something changes."""
    mesh = obj.data
    key = (mesh.name, len(mesh.vertices), len(mesh.loops), mesh.has_custom_normals,
           tuple(round(v, 5) for row in obj.matrix_world for v in row), _cache_version, _shape_signature(obj))
    entry = _hover_cache.get(obj.name)
    if entry is not None and entry[0] == key:
        return entry[1], entry[2]
    mesh.calc_normals_split()
    nloops = len(mesh.loops)
    ln = np.empty(nloops * 3, dtype=np.float32)
    mesh.loops.foreach_get("normal", ln)
    lv = np.empty(nloops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    ln = ln.reshape(-1, 3)
    disp = displayed_local(obj)
    base = basis_local(obj)
    if np.abs(disp - base).max(initial=0.0) > 1e-7:          # shape key: the normals follow the surface
        ps, pt = poly_arrays(mesh)
        ref_v = core.ref_neighbors(lv, ps, pt, len(mesh.vertices))
        fa = core.vertex_frames(base, core.vertex_normals(base, lv, ps, pt), ref_v)
        fb = core.vertex_frames(disp, core.vertex_normals(disp, lv, ps, pt), ref_v)
        ln = core.rotate_frames(ln, fa[lv], fb[lv])
    sums = np.stack([np.bincount(lv, weights=ln[:, k], minlength=len(mesh.vertices)) for k in range(3)], axis=1)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    nrm = core.normalize((sums @ np.linalg.inv(mw[:3, :3])).astype(np.float32))  # inverse transpose
    pos = world_positions(obj, disp)
    _hover_cache[obj.name] = (key, pos, nrm)
    return pos, nrm


def draw_hover_normals(context, cx, cy, radius, settings):
    region, rv3d = context.region, context.region_data
    # cx, cy are in whatever space the current viewport uses. Blender hands draw_cursor the mouse in window
    # pixels and the viewport may be the whole window, so shift into region pixels (and back for drawing).
    vp = gpu.state.viewport_get()
    off = np.array([region.x, region.y], dtype=np.float64) if vp[2] > region.width + 1 or vp[3] > region.height + 1         else np.zeros(2)
    cx, cy = cx - off[0], cy - off[1]
    objs = selected_meshes(context, dedupe=False)
    if not objs or rv3d is None:
        return
    data = [hover_data(ob) for ob in objs]
    pos = np.concatenate([d[0] for d in data])
    nrm = np.concatenate([d[1] for d in data])
    pm = np.array(rv3d.perspective_matrix, dtype=np.float64)
    size = np.array([region.width, region.height], dtype=np.float64)

    def project(p):
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ pm.T
        w = h[:, 3]
        ok = w > 1e-6
        w = np.where(ok, w, 1.0)
        return (h[:, :2] / w[:, None] * 0.5 + 0.5) * size, ok

    p2, ok = project(pos.astype(np.float64))
    near = ok & (np.linalg.norm(p2 - np.array([cx, cy]), axis=1) < radius)
    idx = np.nonzero(near)[0]
    if idx.size == 0:
        return
    cam = np.array(rv3d.view_matrix.inverted().translation)
    facing = np.einsum("ij,ij->i", nrm[idx], cam - pos[idx]) > 0
    idx = idx[facing][:500]
    if idx.size == 0:
        return
    eps = max(settings.blur_distance, 0.01) * 0.5
    tip2, _ = project(pos[idx].astype(np.float64) + nrm[idx] * eps)
    d = tip2 - p2[idx]
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    tip = p2[idx] + d / np.maximum(ln, 1e-6) * 14.0
    coords = np.empty((idx.size * 2, 2), dtype=np.float32)
    coords[0::2] = p2[idx] + off
    coords[1::2] = tip + off
    shader = _flat_shader()
    batch = batch_for_shader(shader, "LINES", {"pos": coords.tolist()})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", (0.3, 0.8, 1.0, 0.9))
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def draw_brush_cursor(context, xy):
    settings = context.scene.emils_normal_brush
    _last_cursor["xy"] = tuple(xy)
    _last_cursor["calls"] += 1
    x, y = xy
    draw_circle(x, y, settings.radius, (1.0, 1.0, 1.0, 0.9))
    draw_circle(x, y, settings.radius * settings.strength, (1.0, 1.0, 1.0, 0.3))
    if settings.show_normals:
        draw_hover_normals(context, x, y, settings.radius, settings)


# ----------------------------------------------------------------- the tool

class EmilsNormalBrushTool(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "emils.normal_brush"
    bl_label = "Emil's Normal Brush"
    bl_description = ("Blur custom normals under the cursor on every selected mesh, also across separate islands and objects.\n"
                     "Ctrl restores, F sets the radius, Shift+F the strength")
    bl_icon = "brush.sculpt.smooth"
    bl_widget = None
    bl_cursor = "CROSSHAIR"
    bl_keymap = (
        ("emils.normal_brush_stroke", {"type": "LEFTMOUSE", "value": "PRESS"},
         {"properties": [("mode", "BLUR")]}),
        ("emils.normal_brush_stroke", {"type": "LEFTMOUSE", "value": "PRESS", "ctrl": True},
         {"properties": [("mode", "RESTORE")]}),
        ("emils.normal_brush_adjust", {"type": "F", "value": "PRESS"},
         {"properties": [("target", "RADIUS")]}),
        ("emils.normal_brush_adjust", {"type": "F", "value": "PRESS", "shift": True},
         {"properties": [("target", "STRENGTH")]}),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        from . import ui
        ui.draw_props(layout, context.scene.emils_normal_brush)

    @staticmethod
    def draw_cursor(context, tool, xy):
        draw_brush_cursor(context, xy)


TOOL_ID = EmilsNormalBrushTool.bl_idname
_previous_tool = {"idname": "builtin.select_box"}   # what the Stop button switches back to


def active_tool_id(context):
    if context.workspace is None:
        return ""
    tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
    return tool.idname if tool else ""


class EMILSNB_OT_toggle(bpy.types.Operator):
    bl_idname = "emils.normal_brush_toggle"
    bl_label = "Start / Stop Normal Brush"
    bl_description = "Turn Emil's Normal Brush on (switches to Object Mode) or back off to the tool you had before"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        if active_tool_id(context) == TOOL_ID:
            bpy.ops.wm.tool_set_by_id(name=_previous_tool["idname"])
            return {"FINISHED"}
        if not selected_meshes(context):
            self.report({"WARNING"}, "Select a mesh first")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        prev = active_tool_id(context)
        if prev and prev != TOOL_ID:
            _previous_tool["idname"] = prev
        bpy.ops.wm.tool_set_by_id(name=TOOL_ID)
        return {"FINISHED"}


classes = (EMILSNB_OT_stroke, EMILSNB_OT_adjust, EMILSNB_OT_auto_distance, EMILSNB_OT_clear, EMILSNB_OT_toggle)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Own slot at the end of the toolbar. With after={"builtin.select_box"} Blender hides it inside the
    # select tools' flyout group, where nobody finds it.
    bpy.utils.register_tool(EmilsNormalBrushTool, separator=True)


def unregister():
    bpy.utils.unregister_tool(EmilsNormalBrushTool)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    _hover_cache.clear()
