# Changelog

## 1.0.0 (2026-11-07)

First version, honestly it just works, which surprised me.

- Normal Brush tool for Object Mode: blur custom normals under the cursor, neighbours found by world distance so it works across separate, unwelded mesh islands.
- Ctrl restores toward the plain vertex normals, F changes the radius, Shift+F the strength.
- Radius, strength, blur distance (with an Auto button), Mirror X and Show Normals, in the tool header and in the sidebar (Emil tab).
- Turns Auto Smooth on (180 degrees) when it's off, warns about modifiers that overwrite custom normals, Clear Custom Normals button.
- One stroke is one undo step, Esc or right click cancels it.
