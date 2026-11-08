# Changelog

## 1.1.0 (2026-11-07)

Works on every selected mesh now, not just the active one. Two separate balls blend fine!

- The brush paints all selected mesh objects (linked duplicates sharing one mesh only count once) and blurs normals across objects, in world space, so a seam between two objects is no different from a seam inside one.
- Restore stays per object, Mirror X mirrors across the active object's local X, only objects whose normals changed get written, one stroke is still one undo step.
- Start Brush needs at least one selected mesh, Auto blur distance uses the mean edge length of all selected meshes, Clear Custom Normals clears all of them, the modifier warning lists `Object: Modifier`, Show Normals draws on every selected mesh.

## 1.0.0 (2026-11-07)

First version, honestly it just works, which surprised me.

- Normal Brush tool for Object Mode: blur custom normals under the cursor, neighbours found by world distance so it works across separate, unwelded mesh islands.
- Ctrl restores toward the plain vertex normals, F changes the radius, Shift+F the strength.
- Radius, strength, blur distance (with an Auto button), Mirror X and Show Normals, in the tool header and in the sidebar (Emil tab).
- Turns Auto Smooth on (180 degrees) when it's off, warns about modifiers that overwrite custom normals, Clear Custom Normals button.
- One stroke is one undo step, Esc or right click cancels it.
