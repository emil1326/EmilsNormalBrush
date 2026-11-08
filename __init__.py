bl_info = {
    "name": "Emil's Normal Brush",
    "author": "Emil",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Toolbar (Object Mode) and Sidebar > Emil",
    "description": "A brush that blurs custom split normals on all selected meshes, also across separate islands and objects",
    "category": "Mesh",
}

if "bpy" in locals():
    import importlib
    for _m in (core, props, brush, ui):
        importlib.reload(_m)

import bpy  # noqa: E402

from . import core, props, brush, ui  # noqa: E402,F401


def register():
    props.register()
    brush.register()
    ui.register()


def unregister():
    ui.unregister()
    brush.unregister()
    props.unregister()


if __name__ == "__main__":
    register()
