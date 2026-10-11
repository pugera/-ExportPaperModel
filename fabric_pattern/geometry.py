"""Intrinsic cloth metrics and topology. NumPy is bundled with Blender."""
import heapq
from collections import defaultdict, deque
import numpy as np


def topology(mesh):
    verts = np.array([v.co[:] for v in mesh.vertices], dtype=float)
    faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)
    edge_ids = {tuple(sorted(e.vertices)): e.index for e in mesh.edges}
    face_edges = np.array([[edge_ids[tuple(sorted((f[j], f[(j+1)%3])))]
                            for j in range(3)] for f in faces], dtype=np.int32)
    edge_faces = [[] for _ in mesh.edges]
    for fi, edges in enumerate(face_edges):
        for ei in edges:
            edge_faces[ei].append(fi)
    if any(len(fs) > 2 for fs in edge_faces):
        raise ValueError('Non-manifold edges: repair edges shared by more than two faces.')
    lengths = np.linalg.norm(verts[faces[:, 1]]-verts[faces[:, 0]], axis=1)
    cross = np.cross(verts[faces[:, 1]]-verts[faces[:, 0]], verts[faces[:, 2]]-verts[faces[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    scale = max(float(np.ptp(verts, axis=0).max()), 1e-12)
    if np.any(area2 < scale*scale*1e-13):
        bad = np.where(area2 < scale*scale*1e-13)[0].tolist()
        raise ValueError('Zero-area or extremely thin triangles: %s' % bad[:12])
    x = np.sum((verts[faces[:, 2]]-verts[faces[:, 0]]) *
               (verts[faces[:, 1]]-verts[faces[:, 0]]), axis=1)/lengths
    rest = np.zeros((len(faces), 2, 2))
    rest[:, 0, 0], rest[:, 0, 1], rest[:, 1, 1] = lengths, x, area2/lengths
    return dict(verts=verts, faces=faces, face_edges=face_edges, edge_faces=edge_faces,
                rest=rest, area=area2/2, centers=verts[faces].mean(axis=1))


def components(t, cuts):
    adjacency = [[] for _ in t['faces']]
    for ei, fs in enumerate(t['edge_faces']):
        if len(fs) == 2 and ei not in cuts:
            a, b = fs
            adjacency[a].append(b)
            adjacency[b].append(a)
    labels = np.full(len(adjacency), -1, dtype=np.int32)
    groups = []
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        group, stack = [], [start]
        labels[start] = len(groups)
        while stack:
            f = stack.pop()
            group.append(f)
            for n in adjacency[f]:
                if labels[n] < 0:
                    labels[n] = labels[start]
                    stack.append(n)
        groups.append(np.array(sorted(group), dtype=np.int32))
    return groups, labels, adjacency


def principal_stretches(rest, uv):
    """Singular values of the physical map FLAT CLOTH -> 3D SURFACE."""
    flat = np.stack((uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]), axis=2)
    det = np.linalg.det(flat)
    valid = np.abs(det) > 1e-20
    inv = np.zeros_like(flat)
    inv[valid] = np.linalg.inv(flat[valid])
    values = np.linalg.svd(rest @ inv, compute_uv=False)
    values[~valid] = (1e15, 1e-15)
    return values, det


def overlap_pairs(tris):
    """Strict positive-area triangle overlaps, including folds; spatial hash + SAT."""
    if len(tris) < 2:
        return []
    lo, hi = tris.min(axis=1), tris.max(axis=1)
    extent = np.maximum(hi.max(axis=0)-lo.min(axis=0), 1e-10)
    cell = max(np.sqrt(float(np.prod(extent))/max(len(tris), 1))*2, extent.max()/256)
    origin, grid, pairs = lo.min(axis=0), defaultdict(list), set()
    for i in range(len(tris)):
        a, b = np.floor((lo[i]-origin)/cell).astype(int), np.floor((hi[i]-origin)/cell).astype(int)
        for x in range(a[0], b[0]+1):
            for y in range(a[1], b[1]+1):
                for j in grid[x, y]:
                    if np.all(np.minimum(hi[i], hi[j])-np.maximum(lo[i], lo[j]) > cell*1e-9):
                        pairs.add((j, i))
                grid[x, y].append(i)
    hits = []
    ordered = sorted(pairs)
    for start in range(0, len(ordered), 4096):
        ij = np.array(ordered[start:start+4096], dtype=int)
        a, b = tris[ij[:, 0]], tris[ij[:, 1]]
        edges = np.concatenate((np.roll(a, -1, axis=1)-a, np.roll(b, -1, axis=1)-b), axis=1)
        axes = np.stack((-edges[:, :, 1], edges[:, :, 0]), axis=2)
        axes /= np.maximum(np.linalg.norm(axes, axis=2, keepdims=True), 1e-30)
        ap, bp = np.einsum('nvi,nai->nav', a, axes), np.einsum('nvi,nai->nav', b, axes)
        depth = np.minimum(ap.max(axis=2), bp.max(axis=2))-np.maximum(ap.min(axis=2), bp.min(axis=2))
        good = np.all(depth > extent.max()*2e-7, axis=1)
        hits.extend(map(tuple, ij[good].tolist()))
    return hits


def measure(t, uv, groups, stretch, compression, check_overlap=True):
    values, det = principal_stretches(t['rest'], uv)
    reports = []
    for group in groups:
        raw = values[group]
        low, high = float(raw.min()), float(raw.max())
        # Minimax utilisation: (high/k-1)/stretch = (1-low/k)/compression.
        if stretch+compression > 0:
            scale = (compression*high+stretch*low)/(stretch+compression)
        else:
            scale = (high+low)/2
        scale = max(scale, 1e-15)
        rates = raw/scale
        max_s, max_c = max(0., float(rates.max())-1), max(0., 1-float(rates.min()))
        flipped = bool(np.any(det[group] > 1e-18) and np.any(det[group] < -1e-18))
        error = max(max_s/max(stretch, 1e-9), max_c/max(compression, 1e-9))
        ok = max_s <= stretch+1e-6 and max_c <= compression+1e-6 and not flipped
        overlaps = overlap_pairs(uv[group]) if ok and check_overlap else []
        if overlaps:
            ok = False
        face_error = np.maximum(np.maximum(rates[:, 0]-1, 0)/max(stretch, 1e-9),
                                np.maximum(1-rates[:, 1], 0)/max(compression, 1e-9))
        reports.append(dict(faces=group, scale=scale, stretch=max_s, compression=max_c,
                            flipped=flipped, overlaps=len(overlaps), ok=ok, error=error,
                            face_error=face_error))
    return reports


def dijkstra(nodes, adjacency, start):
    allowed, dist, parent = set(nodes), {start: 0.}, {}
    heap = [(0., start)]
    while heap:
        d, a = heapq.heappop(heap)
        if d != dist[a]:
            continue
        for b, w in adjacency[a]:
            if b in allowed and d+w < dist.get(b, float('inf')):
                dist[b], parent[b] = d+w, a
                heapq.heappush(heap, (d+w, b))
    return dist, parent


def bisect_chart(t, group, adjacency):
    """Geodesic Voronoi bisection on the face graph; each seed stays connected."""
    if len(group) < 2:
        return set()
    weighted = {int(a): [(b, max(float(np.linalg.norm(t['centers'][a]-t['centers'][b])), 1e-12))
                         for b in adjacency[a]] for a in group}
    d, _ = dijkstra(group, weighted, int(group[0]))
    a = max(d, key=d.get)
    da, _ = dijkstra(group, weighted, a)
    b = max(da, key=da.get)
    db, _ = dijkstra(group, weighted, b)
    side = {int(f): da[int(f)] <= db[int(f)] for f in group}
    return {ei for ei, fs in enumerate(t['edge_faces'])
            if len(fs) == 2 and fs[0] in side and fs[1] in side and side[fs[0]] != side[fs[1]]}


def dart_path(t, group, cuts, mesh, face_error):
    """Shortest internal edge path from a strained interior vertex to a boundary."""
    group_set = set(group.tolist())
    boundary, graph, edge_lookup = set(), defaultdict(list), {}
    vertices = set(t['faces'][group].reshape(-1).tolist())
    for ei, fs in enumerate(t['edge_faces']):
        inside = [f for f in fs if f in group_set]
        if not inside:
            continue
        a, b = mesh.edges[ei].vertices
        if len(inside) == 1 or ei in cuts:
            boundary.update((a, b))
        elif len(inside) == 2:
            w = float(np.linalg.norm(t['verts'][a]-t['verts'][b]))
            graph[a].append((b, w))
            graph[b].append((a, w))
            edge_lookup[tuple(sorted((a, b)))] = ei
    if not boundary:
        return set()
    heap = [(0., v) for v in boundary]
    heapq.heapify(heap)
    dist, parent = {v: 0. for v in boundary}, {}
    while heap:
        d, a = heapq.heappop(heap)
        if d != dist[a]:
            continue
        for b, w in graph[a]:
            if d+w < dist.get(b, float('inf')):
                dist[b], parent[b] = d+w, a
                heapq.heappush(heap, (d+w, b))
    scores = defaultdict(float)
    for f, error in zip(group, face_error):
        for v in t['faces'][f]:
            if v not in boundary and v in dist:
                scores[int(v)] = max(scores[int(v)], float(error)*dist[v]**0.35)
    if not scores:
        return set()
    end = max(scores, key=scores.get)
    path = set()
    while end not in boundary:
        if end not in parent:
            return set()
        nxt = parent[end]
        path.add(edge_lookup[tuple(sorted((end, nxt)))])
        end = nxt
    return path if len(path) >= 2 else set()
