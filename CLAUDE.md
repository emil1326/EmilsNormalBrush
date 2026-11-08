# Emil's Normal Brush

Brush that blurs custom split normals in Object Mode, across unwelded mesh islands (neighbours by world distance). **Blender 3.6 LTS only**, no 4.x/5.x code. README.md explains how to use it.

## Layout

`core.py` pure numpy math (no bpy, the testable part), `brush.py` tool + stroke operator + radius/strength adjuster + cursor drawing, `props.py` `scene.emils_normal_brush`, `ui.py` sidebar panel (tab "Emil"). Add-ons never import each other, keep it that way.

## Dev loop

- Live-linked: `%LOCALAPPDATA%\Packages\BlenderFoundation.Blender3.6LTS_ppwjx1n5r4v9t\LocalCache\Roaming\Blender Foundation\Blender\3.6\scripts\addons\EmilsNormalBrush` is a junction to this folder. Emil enables the add-on himself, never touch his userpref.blend or a GUI Blender he has open. Don't delete the junction recursively.
- `python tests/run_all.py` before every commit (`--headless` / `--gui` for one suite, `-v` for every line). It drives the Store Blender 3.6 by itself, the GUI suite gets its own config with Tablet API = Windows Ink (Huion's wintab32.dll crashes Blender otherwise).
- Release: bump `bl_info["version"]`, CHANGELOG entry, `python tools/build_zip.py` (writes `../dist/EmilsNormalBrush-<version>.zip`).

## Notes

- Dab logic lives in `core.NormalBrushState.dab` / `brush.Session.dab`, the modal only converts mouse to hit + world radius, so tests drive it without events.
- The GUI test can't make Blender call `draw_cursor` (that would need the real mouse), so the cursor code is run through a POST_PIXEL handler. `draw_cursor` gets the mouse in window or region space depending on the viewport, `draw_hover_normals` checks `gpu.state.viewport_get()` for that. If the cursor circle ever sits offset in the real viewport, look there first.
- Writing text (README, CHANGELOG, UI strings) follows Emil's voice from the global CLAUDE.md.
