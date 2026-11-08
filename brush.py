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


def world_positions(obj):
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    return (co @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)


def mesh_edges(mesh):
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", ev)
    return ev.reshape(-1, 2)


def auto_blur_distance(obj):
    return 2.5 * core.mean_edge_length(world_positions(obj), mesh_edges(obj.data))


class Session:
    """Everything one stroke needs: the numpy state, a BVH to raycast, and the write-back."""

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        mesh = self.mesh
        prepare_mesh(mesh)
        mesh.calc_normals_split()
        nloops, nverts = len(mesh.loops), len(mesh.vertices)
        ln = np.empty(nloops * 3, dtype=np.float32)
        mesh.loops.foreach_get("normal", ln)
        lv = np.empty(nloops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", lv)
        pn = np.empty(nverts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("normal", pn)
        self.vw = world_positions(obj)
        self.state = core.NormalBrushState(self.vw, lv, ln.reshape(-1, 3), pn.reshape(-1, 3), obj.matrix_world)
        self.bvh = self._build_bvh()
        self.dirty = False

    def _build_bvh(self):
        mesh = self.mesh
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        return BVHTree.FromPolygons(self.vw.tolist(), tris.reshape(-1, 3).tolist())

    def raycast(self, origin, direction):
        loc, _normal, _index, _dist = self.bvh.ray_cast(origin, direction)
        return loc

    def dab(self, hit_world, world_radius, settings, mode="BLUR"):
        changed = self.state.dab(hit_world, world_radius, settings.strength, settings.blur_distance,
                                 mode, settings.mirror_x)
        self.dirty |= changed
        return changed

    def write(self):
        if not self.dirty:
            return
        self.mesh.normals_split_custom_set(self.state.ln)
        self.mesh.update()
        self.dirty = False

    def restore_start(self):
        self.state.reset()
        self.dirty = True
        self.write()


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
        obj = context.active_object
        if obj is None or obj.type != "MESH" or obj.mode != "OBJECT":
            self.report({"WARNING"}, "Pick a mesh object and stay in Object Mode")
            return {"CANCELLED"}
        if len(obj.data.loops) == 0:
            self.report({"WARNING"}, "The mesh has no faces")
            return {"CANCELLED"}
        self.session = Session(obj)
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
        hit = self.session.raycast(origin, direction)
        self.last_xy = xy
        if hit is None:
            return False
        r = world_radius_at(context, xy, hit, settings.radius)
        return self.session.dab(hit, r, settings, self.mode)

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
    bl_description = "Set the blur distance to about 2.5 times the mean edge length of the active mesh (world space)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        value = auto_blur_distance(context.active_object)
        if value <= 0:
            self.report({"WARNING"}, "No edges to measure")
            return {"CANCELLED"}
        context.scene.emils_normal_brush.blur_distance = value
        return {"FINISHED"}


class EMILSNB_OT_clear(bpy.types.Operator):
    bl_idname = "emils.normal_brush_clear"
    bl_label = "Clear Custom Normals"
    bl_description = "Remove the custom split normals of the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH" and ob.mode == "OBJECT"

    def execute(self, context):
        ob = context.active_object
        with context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
            result = bpy.ops.mesh.customdata_custom_splitnormals_clear()
        bump_cache()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


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


def hover_data(obj):
    """World positions and unit vertex normals (world space) of obj, cached until something changes."""
    mesh = obj.data
    key = (mesh.name, len(mesh.vertices), len(mesh.loops), mesh.has_custom_normals,
           tuple(round(v, 5) for row in obj.matrix_world for v in row), _cache_version)
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
    sums = np.stack([np.bincount(lv, weights=ln[:, k], minlength=len(mesh.vertices)) for k in range(3)], axis=1)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    nrm = core.normalize((sums @ np.linalg.inv(mw[:3, :3])).astype(np.float32))  # inverse transpose
    pos = world_positions(obj)
    _hover_cache[obj.name] = (key, pos, nrm)
    return pos, nrm


def draw_hover_normals(context, cx, cy, radius, settings):
    obj = context.active_object
    region, rv3d = context.region, context.region_data
    # cx, cy are in whatever space the current viewport uses. Blender hands draw_cursor the mouse in window
    # pixels and the viewport may be the whole window, so shift into region pixels (and back for drawing).
    vp = gpu.state.viewport_get()
    off = np.array([region.x, region.y], dtype=np.float64) if vp[2] > region.width + 1 or vp[3] > region.height + 1         else np.zeros(2)
    cx, cy = cx - off[0], cy - off[1]
    if obj is None or obj.type != "MESH" or rv3d is None:
        return
    pos, nrm = hover_data(obj)
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
    bl_description = ("Blur custom normals under the cursor, also across separate mesh islands.\n"
                     "Ctrl restores, F sets the radius, Shift+F the strength")
    bl_icon = "ops.generic.select_circle"
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


classes = (EMILSNB_OT_stroke, EMILSNB_OT_adjust, EMILSNB_OT_auto_distance, EMILSNB_OT_clear)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.utils.register_tool(EmilsNormalBrushTool, after={"builtin.select_box"}, separator=True)


def unregister():
    bpy.utils.unregister_tool(EmilsNormalBrushTool)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    _hover_cache.clear()
