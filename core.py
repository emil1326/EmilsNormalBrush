"""
Emil's Normal Brush, the math. Pure numpy, no bpy, so tests can drive it headless.

Everything here works on per-loop normals (object local space) and per-vertex
positions in world space. Neighbours are found by DISTANCE, not by topology,
which is the whole point: two overlapping islands aren't connected. That also
holds for islands that are separate OBJECTS: the state owns one slice per
object, blurs over all of them in world space and converts the result back
into each object's local space.
"""

import numpy as np

CHUNK = 1500  # affected vertices handled per matrix block, keeps memory small


def normalize(a):
    """Row-wise normalize, zero rows stay zero."""
    length = np.linalg.norm(a, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return a / length


def falloff(dist, radius):
    """Smooth brush falloff: (1 - (d/r)^2)^2, 0 outside the radius."""
    t = np.clip(dist / radius, 0.0, 1.0)
    return (1.0 - t * t) ** 2


def mean_edge_length(world_pos, edge_verts):
    """Mean edge length. edge_verts is (E, 2) of vertex indices."""
    if len(edge_verts) == 0:
        return 0.0
    d = world_pos[edge_verts[:, 0]] - world_pos[edge_verts[:, 1]]
    return float(np.linalg.norm(d, axis=1).mean())


class ObjectNormals:
    """One object's slice: local loop normals, world vertex positions, and its transform."""

    def __init__(self, vert_world, loop_vert, loop_normals, plain_normals, matrix_world=None):
        self.vw = np.ascontiguousarray(vert_world, dtype=np.float32)          # (V,3) world
        self.loop_vert = np.asarray(loop_vert, dtype=np.int64)                # (L,)
        self.ln = np.ascontiguousarray(loop_normals, dtype=np.float32)        # (L,3) local
        self.start_ln = self.ln.copy()
        self.plain = normalize(np.asarray(plain_normals, dtype=np.float32))   # (V,3) local
        self.mw = np.identity(4) if matrix_world is None else np.array(matrix_world, dtype=np.float64)
        self.mw_inv = np.linalg.inv(self.mw)
        m3 = self.mw[:3, :3]
        self.to_world = np.linalg.inv(m3).T.astype(np.float32)   # local normal -> world (inverse transpose)
        self.to_local = m3.T.astype(np.float32)                  # world normal -> local (transpose)
        self.nverts = len(self.vw)
        self.changed = False       # any loop normal moved since the last write
        self.touched = False       # any loop normal moved since the stroke began
        self.refresh()

    def refresh(self):
        """Recompute the local unit vertex normals (mean of the loop normals) and their world version."""
        n, lv = self.nverts, self.loop_vert
        sums = np.stack([np.bincount(lv, weights=self.ln[:, k], minlength=n) for k in range(3)], axis=1)
        self.vn = normalize(sums.astype(np.float32))
        self.vn_world = normalize(self.vn @ self.to_world.T)

    def reset(self):
        self.ln[:] = self.start_ln
        self.refresh()

    def mirror_point(self, hit_world):
        """Mirror a world point across this object's local X plane."""
        local = self.mw_inv @ np.append(np.asarray(hit_world, dtype=np.float64), 1.0)
        local[0] = -local[0]
        return (self.mw @ local)[:3]


class NormalBrushState:
    """All objects of a stroke. With one object this is the plain single-object brush."""

    def __init__(self, objects, mirror_ref=0):
        self.objs = list(objects)
        self.mirror_ref = mirror_ref
        sizes = [o.nverts for o in self.objs]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.vw = np.ascontiguousarray(np.concatenate([o.vw for o in self.objs]), dtype=np.float32)
        self.vn = np.concatenate([o.vn_world for o in self.objs])   # (N,3) world unit vertex normals

    def _sync_vn(self, i):
        o = self.objs[i]
        self.vn[self.offsets[i]:self.offsets[i + 1]] = o.vn_world

    def reset(self):
        for i, o in enumerate(self.objs):
            o.reset()
            self._sync_vn(i)

    def mirror_point(self, hit_world):
        return self.objs[self.mirror_ref].mirror_point(hit_world)

    def dab(self, hit_world, radius, strength, blur_distance, mode="BLUR", mirror_x=False):
        """One brush dab. Returns True when any loop normal changed."""
        changed = self._dab_one(np.asarray(hit_world, dtype=np.float64), radius, strength, blur_distance, mode)
        if mirror_x:
            changed |= self._dab_one(self.mirror_point(hit_world), radius, strength, blur_distance, mode)
        return changed

    def _dab_one(self, hit, radius, strength, blur_distance, mode):
        if radius <= 0 or strength <= 0:
            return False
        rel = self.vw - hit.astype(np.float32)
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        aff = np.nonzero(dist < radius)[0]
        if aff.size == 0:
            return False
        weight = falloff(dist[aff], radius) * strength
        target_world = None if mode == "RESTORE" else self._blur_targets(aff, rel, radius, blur_distance)

        any_changed = False
        for i, o in enumerate(self.objs):
            lo, hi = np.searchsorted(aff, [self.offsets[i], self.offsets[i + 1]])
            if hi == lo:
                continue
            idx = aff[lo:hi] - self.offsets[i]
            if mode == "RESTORE":
                target = o.plain[idx]
            else:
                target = normalize(target_world[lo:hi] @ o.to_local.T)   # local = M^T * world
            wv = np.zeros(o.nverts, dtype=np.float32)
            wv[idx] = weight[lo:hi]
            tv = np.zeros((o.nverts, 3), dtype=np.float32)
            tv[idx] = target
            wl = wv[o.loop_vert]
            sel = np.nonzero(wl > 0)[0]
            if sel.size == 0:
                continue
            w = wl[sel][:, None]
            o.ln[sel] = normalize(o.ln[sel] * (1.0 - w) + tv[o.loop_vert[sel]] * w)
            o.refresh()
            self._sync_vn(i)
            o.changed = o.touched = True
            any_changed = True
        return any_changed

    def _blur_targets(self, aff, rel, radius, blur_distance):
        """Normalized mean (world space) of the vertex normals within blur_distance of each affected vertex."""
        reach = radius + blur_distance
        rr = np.einsum("ij,ij->i", rel, rel)
        cand = np.nonzero(rr < reach * reach)[0]                  # everything that can matter
        cpos = rel[cand]                                          # centered on the hit, float32
        cvn = self.vn[cand]
        csq = np.einsum("ij,ij->i", cpos, cpos)
        bd2 = blur_distance * blur_distance
        out = np.empty((aff.size, 3), dtype=np.float32)
        pos_in_cand = np.searchsorted(cand, aff)
        for start in range(0, aff.size, CHUNK):
            part = pos_in_cand[start:start + CHUNK]
            apos = cpos[part]
            d2 = csq[part][:, None] + csq[None, :] - 2.0 * (apos @ cpos.T)
            out[start:start + CHUNK] = (d2 < bd2).astype(np.float32) @ cvn
        fallback = self.vn[aff]
        length = np.linalg.norm(out, axis=1, keepdims=True)
        bad = length[:, 0] < 1e-8
        out = out / np.maximum(length, 1e-12)
        out[bad] = fallback[bad]
        return out
