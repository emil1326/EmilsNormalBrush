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
FACING_LIMIT = -0.25  # dot above this counts as "same side": a crease of two merged balls passes, the back of a thin wall (~-1) does not
CANCEL_LIMIT = 0.3    # blur target with |sum| under this x the summed weights is cancellation garbage, leave that vertex alone


def normalize(a):
    """Row-wise normalize, zero rows stay zero."""
    length = np.linalg.norm(a, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return a / length


def falloff(dist, radius):
    """Smooth brush falloff: (1 - (d/r)^2)^2, 0 outside the radius."""
    t = np.clip(dist / radius, 0.0, 1.0)
    return (1.0 - t * t) ** 2


def vertex_normals(pos, loop_vert, poly_start, poly_total):
    """Unit vertex normals the way Blender computes them: face normals weighted by the corner angle.
    pos (V,3), loop_vert (L,), poly_start / poly_total (P,) as in mesh.polygons.loop_start / loop_total."""
    pos = np.asarray(pos, dtype=np.float64)
    loop_vert = np.asarray(loop_vert, dtype=np.int64)
    poly_start = np.asarray(poly_start, dtype=np.int64)
    poly_total = np.asarray(poly_total, dtype=np.int64)
    n_loops, n_polys = len(loop_vert), len(poly_start)
    idx = np.arange(n_loops)
    last = poly_start + poly_total - 1
    nxt, prv = idx + 1, idx - 1
    nxt[last] = poly_start
    prv[poly_start] = last
    poly_of = np.repeat(np.arange(n_polys), poly_total)
    p = pos[loop_vert]
    newell = np.cross(p, p[nxt])
    fn = np.stack([np.bincount(poly_of, weights=newell[:, k], minlength=n_polys) for k in range(3)], axis=1)
    fn = normalize(fn)
    e1, e2 = normalize(p[nxt] - p), normalize(p[prv] - p)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", e1, e2), -1.0, 1.0))
    contrib = fn[poly_of] * ang[:, None]
    vn = np.stack([np.bincount(loop_vert, weights=contrib[:, k], minlength=len(pos)) for k in range(3)], axis=1)
    return normalize(vn)


def ref_neighbors(loop_vert, poly_start, poly_total, n_verts):
    """For every vertex the vertex at the other end of the edge leaving its first loop. Blender builds a loop's
    normal space around such an edge, so this is the direction we follow to see how a deformation twists things."""
    loop_vert = np.asarray(loop_vert, dtype=np.int64)
    poly_start = np.asarray(poly_start, dtype=np.int64)
    poly_total = np.asarray(poly_total, dtype=np.int64)
    idx = np.arange(len(loop_vert))
    nxt = idx + 1
    nxt[poly_start + poly_total - 1] = poly_start
    first = np.full(n_verts, len(loop_vert), dtype=np.int64)
    np.minimum.at(first, loop_vert, idx)
    ref = np.arange(n_verts)
    has = first < len(loop_vert)
    ref[has] = loop_vert[nxt[first[has]]]
    return ref


def vertex_frames(pos, normals, ref):
    """Per vertex an orthonormal frame (V,3,3), columns: edge direction projected off the normal, normal x that,
    the normal. Where the edge is degenerate or along the normal any perpendicular will do."""
    pos = np.asarray(pos, dtype=np.float64)
    n = normalize(np.asarray(normals, dtype=np.float64))
    e = pos[ref] - pos
    t = e - n * np.einsum("ij,ij->i", e, n)[:, None]
    bad = np.linalg.norm(t, axis=1) < 1e-9
    if bad.any():
        helper = np.where(np.abs(n[bad][:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
        t[bad] = np.cross(n[bad], helper)
    t = normalize(t)
    return np.stack([t, np.cross(n, t), n], axis=2)


def rotate_frames(v, fa, fb):
    """Row vectors v (n,3) expressed in frame fa (n,3,3) and put back out in frame fb: v' = Fb Fa^T v."""
    v = np.asarray(v, dtype=np.float64)
    local = np.einsum("nji,nj->ni", fa, v)
    return np.einsum("nij,nj->ni", fb, local)


def rotate_arc(v, a, b):
    """Rotate row vectors v (n,3) by the minimal-arc rotation taking unit a (n,3) to unit b (n,3), row by row.
    a == b gives v back, a == -b turns by 180 degrees around some axis perpendicular to a."""
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.einsum("ij,ij->i", a, b)
    k = np.cross(a, b)
    out = np.empty_like(v)
    flip = c < -0.999999
    ok = ~flip
    if ok.any():
        kk, vv, cc = k[ok], v[ok], c[ok]
        # Rodrigues with an unnormalized axis: v c + k x v + k (k.v) / (1 + c)
        out[ok] = (vv * cc[:, None] + np.cross(kk, vv)
                   + kk * (np.einsum("ij,ij->i", kk, vv) / (1.0 + cc))[:, None])
    if flip.any():
        af = a[flip]
        helper = np.where(np.abs(af[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
        u = normalize(np.cross(af, helper))
        vf = v[flip]
        out[flip] = 2.0 * np.einsum("ij,ij->i", u, vf)[:, None] * u - vf
    return out


def mean_edge_length(world_pos, edge_verts):
    """Mean edge length. edge_verts is (E, 2) of vertex indices."""
    if len(edge_verts) == 0:
        return 0.0
    d = world_pos[edge_verts[:, 0]] - world_pos[edge_verts[:, 1]]
    return float(np.linalg.norm(d, axis=1).mean())


class ObjectNormals:
    """One object's slice: local loop normals, world vertex positions, and its transform."""

    def __init__(self, vert_world, loop_vert, loop_normals, plain_normals, matrix_world=None, rot=None):
        self.vw = np.ascontiguousarray(vert_world, dtype=np.float32)          # (V,3) world
        self.loop_vert = np.asarray(loop_vert, dtype=np.int64)                # (L,)
        self.rot = rot            # (basis, displayed) vertex frames when a shape key deforms the mesh, else None
        ln = np.asarray(loop_normals, dtype=np.float64)
        if rot is not None:       # work in displayed space: turn the stored normals along with the surface
            ln = rotate_frames(ln, rot[0][self.loop_vert], rot[1][self.loop_vert])
        self.ln = np.ascontiguousarray(ln, dtype=np.float32)                  # (L,3) local, as displayed
        self.start_ln = self.ln.copy()
        self.plain = normalize(np.asarray(plain_normals, dtype=np.float32))   # (V,3) local
        self.mw = np.identity(4) if matrix_world is None else np.array(matrix_world, dtype=np.float64)
        self.mw_inv = np.linalg.inv(self.mw)
        m3 = self.mw[:3, :3]
        self.to_world = np.linalg.inv(m3).T.astype(np.float32)   # local normal -> world (inverse transpose)
        self.to_local = m3.T.astype(np.float32)                  # world normal -> local (transpose)
        self.plain_world = normalize(self.plain @ self.to_world.T)               # (V,3) world, never changes
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

    def base_normals(self):
        """The loop normals to write into the mesh: back out of displayed space onto the basis shape."""
        if self.rot is None:
            return self.ln
        return rotate_frames(self.ln, self.rot[1][self.loop_vert], self.rot[0][self.loop_vert]).astype(np.float32)

    def mirror_point(self, hit_world):
        """Mirror a world point across this object's local X plane."""
        local = self.mw_inv @ np.append(np.asarray(hit_world, dtype=np.float64), 1.0)
        local[0] = -local[0]
        return (self.mw @ local)[:3]

    def mirror_normal(self, n_world):
        """Mirror a world normal across this object's local X plane."""
        local = self.to_local.astype(np.float64) @ np.asarray(n_world, dtype=np.float64)
        local[0] = -local[0]
        out = self.to_world.astype(np.float64) @ local
        length = np.linalg.norm(out)
        return out / length if length > 1e-12 else out


class NormalBrushState:
    """All objects of a stroke. With one object this is the plain single-object brush."""

    def __init__(self, objects, mirror_ref=0, tris=None, ghost_vw=None):
        """tris: (T,3) vertex indices into the concatenated world positions (all objects), the static faces.
        ghost_vw: (G,3) extra world positions (a Mirror modifier's copies) that tris may index after the real ones.
        They count as faces for the blur but are never painted."""
        self.objs = list(objects)
        self.mirror_ref = mirror_ref
        sizes = [o.nverts for o in self.objs]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.vw = np.ascontiguousarray(np.concatenate([o.vw for o in self.objs]), dtype=np.float32)
        self.vn = np.concatenate([o.vn_world for o in self.objs])   # (N,3) world unit vertex normals
        self.plain_w = np.concatenate([o.plain_world for o in self.objs])   # (N,3) world plain vertex normals
        allw = self.vw if ghost_vw is None else np.concatenate([self.vw, np.asarray(ghost_vw, dtype=np.float32)])
        self._setup_tris(np.zeros((0, 3), dtype=np.int64) if tris is None else np.asarray(tris, dtype=np.int64), allw)

    def _setup_tris(self, tris, allw):
        """Static per-triangle data (world space): corners, area-weighted normal, centroid, bounding radius."""
        a, b, c = (allw[tris[:, k]].astype(np.float64) for k in range(3))
        cr = np.cross(b - a, c - a)
        area2 = np.linalg.norm(cr, axis=1)
        keep = area2 > 1e-14                                        # drop degenerate triangles
        a, b, c, cr = a[keep], b[keep], c[keep], cr[keep]
        self.tri_a = a.astype(np.float32)
        self.tri_b = b.astype(np.float32)
        self.tri_c3 = c.astype(np.float32)
        self.tri_w = (0.5 * cr).astype(np.float32)                  # area * unit face normal
        self.tri_n = normalize(cr).astype(np.float32)               # unit face normal
        self.tri_wl = (0.5 * np.linalg.norm(cr, axis=1)).astype(np.float32)   # area
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

    def dab(self, hit_world, radius, strength, blur_distance, mode="BLUR", mirror_x=False, hit_normal=None,
            visible=None):
        """One brush dab. hit_normal (world, the face under the cursor) keeps the far side of a thin mesh out of it,
        visible(points, radius) -> bool mask drops vertices hidden behind other faces (the mirrored dab skips that,
        it paints the symmetric spot even if you can't see it).
        Returns True when any loop normal changed."""
        hn = None if hit_normal is None else np.asarray(hit_normal, dtype=np.float64)
        changed = self._dab_one(np.asarray(hit_world, dtype=np.float64), hn, radius, strength, blur_distance, mode,
                                visible)
        if mirror_x:
            mhn = None if hn is None else self.objs[self.mirror_ref].mirror_normal(hn)
            changed |= self._dab_one(self.mirror_point(hit_world), mhn, radius, strength, blur_distance, mode, None)
        return changed

    def _dab_one(self, hit, hit_normal, radius, strength, blur_distance, mode, visible=None):
        if radius <= 0 or strength <= 0:
            return False
        rel = self.vw - hit.astype(np.float32)
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        near = dist < radius
        if hit_normal is not None:                                  # front side only
            near &= self.plain_w @ hit_normal.astype(np.float32) > FACING_LIMIT
        aff = np.nonzero(near)[0]
        if aff.size and visible is not None:                        # nothing hidden behind another face
            aff = aff[visible(self.vw[aff], radius)]
        if aff.size == 0:
            return False
        if mode == "RESTORE":
            target_world = None
        else:
            target_world, valid = self._blur_targets(aff, rel, hit, radius, blur_distance)
            if not valid.all():                                     # cancelled out, leave those vertices alone
                aff, target_world = aff[valid], target_world[valid]
                if aff.size == 0:
                    return False
        weight = falloff(dist[aff], radius) * strength

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
        valid = np.ones(aff.size, dtype=bool)
        if len(self.tri_a) == 0:
            return out, valid
        hit32 = hit.astype(np.float32)
        cen_rel = self.tri_c - hit32                               # centroids relative to the hit
        cen_rel_sq = np.einsum("ij,ij->i", cen_rel, cen_rel)
        reach = radius + blur_distance + self.tri_r                # a triangle can matter if its centroid is this close
        cand = np.nonzero(cen_rel_sq < reach * reach)[0]
        if cand.size == 0:
            return out, valid
        cand = cand[np.argsort(cen_rel[cand, 0], kind="stable")]   # sorted along x, so a chunk only meets a slab
        ca, cb, cc = (x[cand] - hit32 for x in (self.tri_a, self.tri_b, self.tri_c3))
        cw = self.tri_w[cand]
        cn = self.tri_n[cand]
        cwl = self.tri_wl[cand]                                    # areas, for the cancellation test
        cen_rel = cen_rel[cand]
        crad = self.tri_r[cand] + blur_distance
        slack = cen_rel_sq[cand] - crad * crad                     # |c|^2 - R^2, the part of the test that is per triangle
        cx, max_r = cen_rel[:, 0], float(crad.max())
        order = np.argsort(rel[aff, 0], kind="stable")             # affected vertices sorted along x as well
        apos = rel[aff][order]
        apn = self.plain_w[aff][order]
        acc = np.zeros((aff.size, 3), dtype=np.float32)
        wsum = np.zeros(aff.size, dtype=np.float64)
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
            front = dot(cn[ti], apn[start:start + CHUNK][vi]) > FACING_LIMIT   # not the faces on the far side
            vi, ti = vi[front], ti[front]
            if vi.size == 0:
                continue
            dist = np.sqrt(tri_dist2(p[vi], ca[ti], cb[ti], cc[ti]))
            w = falloff(dist, blur_distance)
            for k in range(3):
                acc[start:start + CHUNK, k] = np.bincount(vi, weights=w * cw[ti, k], minlength=p.shape[0])
            wsum[start:start + CHUNK] = np.bincount(vi, weights=w * cwl[ti], minlength=p.shape[0])
        inv = np.argsort(order)                                    # back to the order of `aff`
        acc, wsum = acc[inv], wsum[inv]
        length = np.linalg.norm(acc, axis=1)
        ok = length > 1e-12
        out[ok] = (acc[ok] / length[ok, None]).astype(np.float32)
        valid = length >= CANCEL_LIMIT * wsum                      # wsum 0 (nothing in reach) stays valid, keeps the fallback
        return out, valid


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
