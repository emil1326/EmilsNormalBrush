import bpy

from . import brush


def overwriting_modifiers(obj):
    """Names of modifiers on obj that will replace the custom normals we paint."""
    found = []
    if obj is None:
        return found
    for mod in obj.modifiers:
        if mod.type in {"WEIGHTED_NORMAL", "NORMAL_EDIT"}:
            found.append(mod.name)
        elif mod.type == "DATA_TRANSFER" and mod.use_loop_data and "CUSTOM_NORMAL" in mod.data_types_loops:
            found.append(mod.name)
    return found


def overwriting_modifiers_all(objs):
    """'Object: Modifier' for every overwriting modifier on the given objects."""
    return [f"{ob.name}: {name}" for ob in objs for name in overwriting_modifiers(ob)]


def draw_props(layout, settings):
    layout.prop(settings, "radius")
    layout.prop(settings, "strength")
    row = layout.row(align=True)
    row.prop(settings, "blur_distance")
    row.operator("emils.normal_brush_auto_distance", text="Auto")
    row = layout.row(align=True)
    row.prop(settings, "mirror_x", toggle=True)
    row.prop(settings, "only_visible", toggle=True)
    row.prop(settings, "show_normals", toggle=True)


class EMILSNB_PT_main(bpy.types.Panel):
    bl_label = "Emil's Normal Brush"
    bl_idname = "EMILSNB_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Emil"

    def draw(self, context):
        layout = self.layout
        on = brush.active_tool_id(context) == brush.TOOL_ID
        row = layout.row()
        row.scale_y = 1.6
        row.operator("emils.normal_brush_toggle", text="Stop Brush" if on else "Start Brush",
                     icon="PAUSE" if on else "PLAY", depress=on)
        draw_props(layout, context.scene.emils_normal_brush)
        for name in overwriting_modifiers_all(brush.selected_meshes(context, dedupe=False)):
            layout.label(text=f"{name} overwrites custom normals", icon="ERROR")
        layout.separator()
        layout.operator("emils.normal_brush_clear", text="Clear Custom Normals")


classes = (EMILSNB_PT_main,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
