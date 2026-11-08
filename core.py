"""
Emil's Normal Brush, the math. Pure numpy, no bpy, so tests can drive it headless.

Everything here works on per-loop normals (object local space) and per-vertex
positions in world space. Neighbours are found by DISTANCE, not by topology,
which is the whole point: two overlapping islands aren't connected.
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


class NormalBrushState:
    def __init__(self, vert_world, loop_vert, loop_normals, plain_normals, matrix_world=None):
        self.vw = np.ascontiguousarray(vert_world, dtype=np.float32)          # (V,3) world
        self.loop_vert = np.asarray(loop_vert, dtype=np.int64)                # (L,)
        self.ln = np.ascontiguousarray(loop_normals, dtype=np.float32)        # (L,3) local
        self.start_ln = self.ln.copy()
        self.plain = normalize(np.asarray(plain_normals, dtype=np.float32))   # (V,3) local
        self.mw = np.identity(4) if matrix_world is None else np.array(matrix_world, dtype=np.float64)
        self.mw_inv = np.linalg.inv(self.mw)
        self.nverts = len(self.vw)
        self._refresh_vertex_normals()

    def _refresh_vertex_normals(self):
        n = self.nverts
        lv = self.loop_vert
        sums = np.stack([np.bincount(lv, weights=self.ln[:, k], minlength=n) for k in range(3)], axis=1)
        self.vn = normalize(sums.astype(np.float32))

    def reset(self):
        self.ln[:] = self.start_ln
        self._refresh_vertex_normals()

    def mirror_point(self, hit_world):
        """Mirror a world point across the object's local X plane."""
        local = self.mw_inv @ np.append(np.asarray(hit_world, dtype=np.float64), 1.0)
        local[0] = -local[0]
        return (self.mw @ local)[:3]

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
        if mode == "RESTORE":
            target = self.plain[aff]
        else:
            target = self._blur_targets(aff, rel, radius, blur_distance)

        wv = np.zeros(self.nverts, dtype=np.float32)
        wv[aff] = weight
        tv = np.zeros((self.nverts, 3), dtype=np.float32)
        tv[aff] = target
        wl = wv[self.loop_vert]
        sel = np.nonzero(wl > 0)[0]
        if sel.size == 0:
            return False
        w = wl[sel][:, None]
        new = self.ln[sel] * (1.0 - w) + tv[self.loop_vert[sel]] * w
        self.ln[sel] = normalize(new)
        self._refresh_vertex_normals()
        return True

    def _blur_targets(self, aff, rel, radius, blur_distance):
        """Normalized mean of the vertex normals within blur_distance of each affected vertex."""
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
