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

CHUNK = 256  # affected vertices handled per matrix block, keeps memory small


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

    def __init__(self, objects, mirror_ref=0, tris=None):
        """tris: (T,3) vertex indices into the concatenated world positions (all objects), the static faces."""
        self.objs = list(objects)
        self.mirror_ref = mirror_ref
        sizes = [o.nverts for o in self.objs]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.vw = np.ascontiguousarray(np.concatenate([o.vw for o in self.objs]), dtype=np.float32)
        self.vn = np.concatenate([o.vn_world for o in self.objs])   # (N,3) world unit vertex normals
        self._setup_tris(np.zeros((0, 3), dtype=np.int64) if tris is None else np.asarray(tris, dtype=np.int64))

    def _setup_tris(self, tris):
        """Static per-triangle data (world space): corners, area-weighted normal, centroid, bounding radius."""
        a, b, c = (self.vw[tris[:, k]].astype(np.float64) for k in range(3))
        cr = np.cross(b - a, c - a)
        area2 = np.linalg.norm(cr, axis=1)
        keep = area2 > 1e-14                                        # drop degenerate triangles
        a, b, c, cr = a[keep], b[keep], c[keep], cr[keep]
        self.tri_a = a.astype(np.float32)
        self.tri_b = b.astype(np.float32)
        self.tri_c3 = c.astype(np.float32)
        self.tri_w = (0.5 * cr).astype(np.float32)                  # area * unit face normal
        cen = (a + b + c) / 3.0
        self.tri_c = cen.astype(np.float32)
        self.tri_r = np.maximum.reduce([np.linalg.norm(x - cen, axis=1) for x in (a, b, c)]).astype(np.float32)

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
        target_world = None if mode == "RESTORE" else self._blur_targets(aff, rel, hit, radius, blur_distance)

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

    def _blur_targets(self, aff, rel, hit, radius, blur_distance):
        """Normalized sum (world space) of area * face normal * falloff(distance to the triangle / blur_distance)
        over every triangle near each affected vertex. Needs nothing but the static triangles, never the current
        normals, so repeated dabs settle on the surface shape instead of averaging themselves into mush."""
        out = self.vn[aff].copy()                                  # fallback: no face in reach
        if len(self.tri_a) == 0:
            return out
        hit32 = hit.astype(np.float32)
        cen_rel = self.tri_c - hit32                               # centroids relative to the hit
        cen_rel_sq = np.einsum("ij,ij->i", cen_rel, cen_rel)
        reach = radius + blur_distance + self.tri_r                # a triangle can matter if its centroid is this close
        cand = np.nonzero(cen_rel_sq < reach * reach)[0]
        if cand.size == 0:
            return out
        cand = cand[np.argsort(cen_rel[cand, 0], kind="stable")]   # sorted along x, so a chunk only meets a slab
        ca, cb, cc = (x[cand] - hit32 for x in (self.tri_a, self.tri_b, self.tri_c3))
        cw = self.tri_w[cand]
        cen_rel = cen_rel[cand]
        crad = self.tri_r[cand] + blur_distance
        slack = cen_rel_sq[cand] - crad * crad                     # |c|^2 - R^2, the part of the test that is per triangle
        cx, max_r = cen_rel[:, 0], float(crad.max())
        order = np.argsort(rel[aff, 0], kind="stable")             # affected vertices sorted along x as well
        apos = rel[aff][order]
        acc = np.zeros((aff.size, 3), dtype=np.float32)
        for start in range(0, aff.size, CHUNK):
            p = apos[start:start + CHUNK]
            lo, hi = np.searchsorted(cx, [p[0, 0] - max_r, p[-1, 0] + max_r])
            if hi <= lo:
                continue
            # |p - c|^2 < R^2  <=>  -2 p.c + (|c|^2 - R^2) < -|p|^2   (cheap prefilter, sparse vertex/triangle pairs)
            m = (-2.0 * p) @ cen_rel[lo:hi].T
            m += slack[lo:hi]
            vi, ti = np.nonzero(m < -np.einsum("ij,ij->i", p, p)[:, None])
            if vi.size == 0:
                continue
            ti += lo
            dist = np.sqrt(tri_dist2(p[vi], ca[ti], cb[ti], cc[ti]))
            w = falloff(dist, blur_distance)
            for k in range(3):
                acc[start:start + CHUNK, k] = np.bincount(vi, weights=w * cw[ti, k], minlength=p.shape[0])
        acc = acc[np.argsort(order)]                               # back to the order of `aff`
        length = np.linalg.norm(acc, axis=1)
        ok = length > 1e-12
        out[ok] = (acc[ok] / length[ok, None]).astype(np.float32)
        return out


def dot(x, y):
    return np.einsum("ij,ij->i", x, y)


def tri_dist2(p, a, b, c):
    """Squared distance from points p (n,3) to triangles a,b,c (n,3 each), exact (Ericson, Real-Time Collision
    Detection 5.1.5): vertex, edge and face regions, all vectorized. Triangles must not be degenerate."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def safe(num, den):
        return num / np.where(np.abs(den) < 1e-300, 1.0, den)

    denom = va + vb + vc
    u = safe(vb, denom)                                                               # inside the face: q = a + u ab + v ac
    v = safe(vc, denom)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)                               # edge BC
    w = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    u, v = np.where(m, 1.0 - w, u), np.where(m, w, v)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                                             # edge AC
    u, v = np.where(m, 0.0, u), np.where(m, safe(d2, d2 - d6), v)
    m = (d6 >= 0) & (d5 <= d6)                                                        # vertex C
    u, v = np.where(m, 0.0, u), np.where(m, 1.0, v)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                                             # edge AB
    u, v = np.where(m, safe(d1, d1 - d3), u), np.where(m, 0.0, v)
    m = (d3 >= 0) & (d4 <= d3)                                                        # vertex B
    u, v = np.where(m, 1.0, u), np.where(m, 0.0, v)
    m = (d1 <= 0) & (d2 <= 0)                                                         # vertex A
    u, v = np.where(m, 0.0, u), np.where(m, 0.0, v)
    q = a + ab * u[:, None] + ac * v[:, None]
    d = p - q
    return dot(d, d)
