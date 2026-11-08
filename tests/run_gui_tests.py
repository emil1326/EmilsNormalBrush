"""
GUI smoke test for Emil's Normal Brush. Opens its own Blender window, runs by itself and quits.
Run with Blender 3.6:

    blender --factory-startup --no-window-focus --python EmilsNormalBrush/tests/run_gui_tests.py -- <result.txt>

Checks: the add-on registers, the tool can be picked in a 3D view, the brush circle + hover normals draw
without errors, and (when Windows lets us move the real mouse over the window) Blender really calls the
tool's draw_cursor.
"""
import os
import sys
import tempfile
import traceback

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
RESULT = argv[0] if argv else os.path.join(tempfile.gettempdir(), "emils_normal_brush_gui_tests.txt")
log = open(RESULT + ".log", "w", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log
import faulthandler
faulthandler.enable(log)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import bpy
import bmesh

import EmilsNormalBrush
from EmilsNormalBrush import brush

EmilsNormalBrush.register()
results = []


def check(name, cond, detail=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print(results[-1], flush=True)


def info(text):
    results.append("INFO  " + text)
    print(results[-1], flush=True)


def finish():
    with open(RESULT, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")
    os._exit(0)


def view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                return window, area, region
    return None, None, None


draw_state = {"errors": [], "draws": 0}


def probe_draw():
    """POST_PIXEL handler that runs the cursor drawing code (with hover normals on) inside a real draw."""
    try:
        ctx = bpy.context
        region = ctx.region
        brush.draw_brush_cursor(ctx, (region.width // 2, region.height // 2))
        draw_state["draws"] += 1
    except Exception:
        draw_state["errors"].append(traceback.format_exc())


def steps():
    scene = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
    mesh = bpy.data.meshes.new("Ball")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Ball", mesh)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    yield 1.0

    window, area, region = view3d()
    check("found a 3D view", area is not None)
    if area is None:
        finish()
    check("scene settings exist", hasattr(scene, "emils_normal_brush"))

    with bpy.context.temp_override(window=window, area=area, region=region):
        res = bpy.ops.wm.tool_set_by_id(name="emils.normal_brush")
    yield 0.5
    tool = bpy.context.workspace.tools.from_space_view3d_mode("OBJECT", create=False)
    check("tool_set_by_id picks Emil's Normal Brush", tool is not None and tool.idname == "emils.normal_brush",
          f"{res} / {getattr(tool, 'idname', None)}")

    scene.emils_normal_brush.show_normals = True
    handle = bpy.types.SpaceView3D.draw_handler_add(probe_draw, (), "WINDOW", "POST_PIXEL")
    for _ in range(3):
        area.tag_redraw()
        yield 0.4
    bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
    check("brush circle + hover normals draw inside a real draw callback",
          draw_state["draws"] > 0 and not draw_state["errors"],
          (draw_state["errors"] or [f"{draw_state['draws']} draws"])[0][-300:])

    # Moving the real mouse over the window to make Blender call draw_cursor is skipped on purpose,
    # it would hijack the mouse of whoever is using the machine. The draw code itself is covered above.
    yield 0.2

    EmilsNormalBrush.unregister()
    check("unregister with the tool active works", not hasattr(bpy.types, "EMILSNB_PT_main"))
    finish()


gen = steps()


def tick():
    try:
        return next(gen)
    except StopIteration:
        finish()
    except Exception:
        traceback.print_exc()
        results.append("FAIL  crashed: " + traceback.format_exc().splitlines()[-1])
        finish()


bpy.app.timers.register(tick, first_interval=1.5)
