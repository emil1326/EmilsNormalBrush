import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty


class EmilsNormalBrushSettings(bpy.types.PropertyGroup):
    radius: IntProperty(
        name="Radius", description="Brush radius in pixels (F in the viewport)",
        default=60, min=5, max=1000, subtype="PIXEL")
    strength: FloatProperty(
        name="Strength", description="How hard each dab pulls the normals (Shift+F in the viewport)",
        default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    blur_distance: FloatProperty(
        name="Blur Distance",
        description="How far a vertex looks for neighbours, in world units. This is what reaches across the gap between two separate islands",
        default=0.1, min=0.0001, soft_max=10.0, precision=4, subtype="DISTANCE")
    mirror_x: BoolProperty(
        name="Mirror X", description="Also dab at the mirrored spot across the object's local X", default=False)
    show_normals: BoolProperty(
        name="Show Normals", description="Draw the normals of the vertices under the brush while hovering",
        default=False)


classes = (EmilsNormalBrushSettings,)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.emils_normal_brush = PointerProperty(type=EmilsNormalBrushSettings)


def unregister():
    del bpy.types.Scene.emils_normal_brush
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
