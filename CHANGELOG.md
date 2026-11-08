# Changelog

## 1.3.1 (2026-11-07)

The occlusion is a toggle now, **Only Visible**, on by default. Turn it off when you actually want to paint what's hidden behind, like the feather layers underneath.

## 1.3.0 (2026-11-07)

It paints what you actually SEE now. Feathers with a shape key and a Mirror modifier were the test case, and honestly the old version was painting somewhere in the past, where the feather used to be xD

- Shape keys: the brush uses the shape key mix you see (values, relative keys, vertex groups, muted keys, "show only active shape key"), for the raycast, the faces and the vertices. The stored normals get turned along with the surface and turned back when they're written, checked against what Blender itself shows (mean error 0.3 degrees, the odd loop around 5).
- Mirror modifier: click on the mirrored half and the dab lands on the real vertices, and the mirrored faces count for the blur, so the seam at the mirror plane blends with its other half. Several axes, a mirror object and two Mirror modifiers in a row work, bisect and flip are ignored. Blender 3.6 mirrors custom normals itself, so you only paint one side. Mirror X in the brush is a separate thing and stays.
- Occlusion: vertices hidden behind another face (stacked layers, feathers) don't get painted anymore, only what the viewer can see. Mirror X skips that check on purpose, it paints the symmetric spot even if it's behind the mesh.

## 1.2.1 (2026-11-07)

Fixed the "puts my normals in a completely wrong place" bug. Turns out the brush was picking BOTH sides of a mesh when the brush was wider than the mesh is thick, and the blur then averaged faces pointing in opposite directions. They cancel out, the sum points wherever, yeah. Sadly xD

- Only vertices on the side you're pointing at get painted (their normal has to face about the same way as the face under the cursor, `FACING_LIMIT`). Mirror X mirrors that face normal too. Restore follows the same rule.
- The blur target ignores faces that face away from the vertex, so the other side of a thin wall doesn't leak in, but the crease between two merged balls still counts.
- If the faces around a vertex still cancel out, that vertex just stays as it is instead of getting garbage.
- Objects with a negative scale (mirrored) had their face normals computed inside out, now fixed too.

## 1.2.0 (2026-11-07)

Works on low-poly meshes now. With few vertices there was barely anything in reach, so the blur almost didn't happen.

- The blur target of a vertex is now the normalized sum of area * face normal * falloff over every face (of all painted objects, world space) within Blur Distance, measured to the closest point of the face. Faces the vertex belongs to always count fully.
- It no longer reads the current normals, so repeated dabs converge to the surface shape instead of mush. Restore is unchanged.
- Old vertex-normal averaging is gone, no mode switch.
- Auto Blur Distance is 1.5 x the mean edge length now (was 2.5), picked from a sweep on low-poly and dense test meshes.

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
