# Emil's Normal Brush

A brush that blurs custom normals right under your cursor. I had a mesh that's basically two balls stuck together, and there's always that ugly shading line right where they meet, smooth shading or not. So yeah, now I just paint over the seam and the normals line up across both sides, even when the two parts aren't connected at all :>

Built for **Blender 3.6 LTS**.

## Install

Run `python tools/build_zip.py` (or zip the `EmilsNormalBrush` folder), then Edit > Preferences > Add-ons > Install and turn on **Emil's Normal Brush**. The tool shows up in the toolbar in Object Mode (the one with the circle icon, under the select tools), and the settings are also in the 3D View sidebar, in the **Emil** tab.

## Using it

Pick the tool, select your mesh (it has to be the active object), and paint over the seam. Each dab looks at all the vertices within **Blur Distance** of a vertex, wherever they are, connected or not, and pulls that vertex's normals toward their average. That's the whole trick: the distance is world units, so it reaches across the gap between two islands. Hit the **Auto** button next to it and it picks about 2.5 times the mean edge length, which was a good start on everything I tried. Make it bigger for a softer blend, smaller if it starts eating details.

**Strength** is how far each dab pulls, **Radius** is the brush size on screen in pixels. Ctrl+drag does the opposite job, it pulls the normals back toward the plain vertex normals, so you can undo a blur you went overboard with. Mirror X also dabs on the other side of the object's local X, and Show Normals draws little blue lines for the vertices under the circle so you can see what's going on.

Custom normals in 3.6 need Auto Smooth. If it's off on your mesh the brush turns it on with the angle at 180 degrees (so the look stays smooth), if it's already on it leaves your angle alone. Also shade the mesh smooth first, flat faces ignore custom normals. If the panel shows a warning about a modifier (Weighted Normal, Normal Edit, Data Transfer with custom normals), that modifier will overwrite what you paint. **Clear Custom Normals** throws it all away, in case you want to start over.

One stroke is one undo step, and Esc or right click during a stroke puts everything back how it was.

### Controls

- LMB drag: blur
- Ctrl + LMB drag: restore
- F: change the radius (move the mouse left/right, click or Enter to confirm, Esc cancels)
- Shift + F: same for the strength

## Things to know

It ignores modifiers, the brush works on the base mesh. The distances are in world space but the normals live in object space, so it assumes the object has roughly uniform scale. Squash it flat and the blur gets a bit weird. There's no tablet pressure either, I don't use my tablet anymore xD
