"""
Polyhedron Clipping (Sutherland-Hodgman 3D)
Initialize cubes and clip against power planes
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import (
    pos, rad, nbr_ct, nbr_idx,
    verts, faces_begin, faces_count, faces_normal, idx,
    v_ct, f_ct, i_ct, overflow,
    plane_n, plane_b
)

# =============================================================================
# Cube Initialization
# =============================================================================

@ti.kernel
def init_cubes():
    """
    Initialize each particle's polyhedron as an axis-aligned cube.
    
    Cube is centered at particle position with half-extent R_sec.
    Vertices are stored relative to particle position for numerical stability.
    
    Cube structure:
        8 vertices (corners)
        6 faces (quads, stored as CCW loops)
        12 triangles (2 per face, stored as triangle fans in idx[])
    
    Vertex ordering (standard cube):
        v0 = (-R, -R, -R)    v4 = (-R, -R, +R)
        v1 = (+R, -R, -R)    v5 = (+R, -R, +R)
        v2 = (+R, +R, -R)    v6 = (+R, +R, +R)
        v3 = (-R, +R, -R)    v7 = (-R, +R, +R)
    
    Face ordering (normals point outward):
        Face 0: -Z (v0,v1,v2,v3)  normal = (0,0,-1)
        Face 1: +Z (v4,v5,v6,v7)  normal = (0,0,+1)
        Face 2: -Y (v0,v1,v5,v4)  normal = (0,-1,0)
        Face 3: +Y (v3,v2,v6,v7)  normal = (0,+1,0)
        Face 4: -X (v0,v3,v7,v4)  normal = (-1,0,0)
        Face 5: +X (v1,v2,v6,v5)  normal = (+1,0,0)
    """
    R = C.R_SEC
    
    for i in range(C.N):
        # Reset counters
        v_ct[i] = 8
        f_ct[i] = 6
        i_ct[i] = 24  # 6 faces × 4 vertices each
        overflow[i] = 0
        
        # Define 8 cube corners (relative to particle position)
        verts[i, 0] = ti.Vector([-R, -R, -R])
        verts[i, 1] = ti.Vector([+R, -R, -R])
        verts[i, 2] = ti.Vector([+R, +R, -R])
        verts[i, 3] = ti.Vector([-R, +R, -R])
        verts[i, 4] = ti.Vector([-R, -R, +R])
        verts[i, 5] = ti.Vector([+R, -R, +R])
        verts[i, 6] = ti.Vector([+R, +R, +R])
        verts[i, 7] = ti.Vector([-R, +R, +R])
        
        # Define 6 faces (each as a quad, stored as index range)
        # Face 0: -Z (v0,v1,v2,v3)
        faces_begin[i, 0] = 0
        faces_count[i, 0] = 4
        faces_normal[i, 0] = ti.Vector([0.0, 0.0, -1.0])
        idx[i, 0] = 0
        idx[i, 1] = 1
        idx[i, 2] = 2
        idx[i, 3] = 3
        
        # Face 1: +Z (v4,v5,v6,v7)
        faces_begin[i, 1] = 4
        faces_count[i, 1] = 4
        faces_normal[i, 1] = ti.Vector([0.0, 0.0, +1.0])
        idx[i, 4] = 4
        idx[i, 5] = 5
        idx[i, 6] = 6
        idx[i, 7] = 7
        
        # Face 2: -Y (v0,v1,v5,v4)
        faces_begin[i, 2] = 8
        faces_count[i, 2] = 4
        faces_normal[i, 2] = ti.Vector([0.0, -1.0, 0.0])
        idx[i, 8] = 0
        idx[i, 9] = 1
        idx[i, 10] = 5
        idx[i, 11] = 4
        
        # Face 3: +Y (v3,v2,v6,v7)
        faces_begin[i, 3] = 12
        faces_count[i, 3] = 4
        faces_normal[i, 3] = ti.Vector([0.0, +1.0, 0.0])
        idx[i, 12] = 3
        idx[i, 13] = 2
        idx[i, 14] = 6
        idx[i, 15] = 7
        
        # Face 4: -X (v0,v3,v7,v4)
        faces_begin[i, 4] = 16
        faces_count[i, 4] = 4
        faces_normal[i, 4] = ti.Vector([-1.0, 0.0, 0.0])
        idx[i, 16] = 0
        idx[i, 17] = 3
        idx[i, 18] = 7
        idx[i, 19] = 4
        
        # Face 5: +X (v1,v2,v6,v5)
        faces_begin[i, 5] = 20
        faces_count[i, 5] = 4
        faces_normal[i, 5] = ti.Vector([+1.0, 0.0, 0.0])
        idx[i, 20] = 1
        idx[i, 21] = 2
        idx[i, 22] = 6
        idx[i, 23] = 5


# =============================================================================
# Sutherland-Hodgman 3D Clipping
# =============================================================================

# Tolerances
EPS_PLANE = 1e-7  # signed distance tolerance
EPS_MERGE = 1e-6  # vertex deduplication tolerance
A_MIN = 1e-8      # minimum face area (cull tiny faces)

@ti.func
def signed_dist(n: ti.template(), b: ti.f32, p_i: ti.template(), v: ti.template()) -> ti.f32:
    """
    Compute signed distance from vertex v to plane.
    Plane equation: n·(x - p_i) = b
    """
    return n.dot(v) - b  # v is already relative to p_i


@ti.func
def dedup_and_add_vertex(i: ti.i32, v: ti.template()) -> ti.i32:
    """
    Add vertex v to particle i's vertex list, deduplicating if needed.
    Returns vertex index.
    """
    # Linear search for duplicate (fine for small vertex counts)
    found_idx = -1
    for k in range(v_ct[i]):
        if found_idx < 0:  # only check if not found yet
            diff = verts[i, k] - v
            if diff.norm() < EPS_MERGE:
                found_idx = k
    
    # If found duplicate, return it
    result = -1
    if found_idx >= 0:
        result = found_idx
    else:
        # Add new vertex
        if v_ct[i] >= C.V_MAX:
            overflow[i] = 1
            result = -1
        else:
            idx_new = v_ct[i]
            verts[i, idx_new] = v
            v_ct[i] += 1
            result = idx_new
    
    return result


@ti.func
def plane_frame(n: ti.template()) -> ti.types.vector(2, ti.f32):
    """
    Build orthonormal frame (t, u) on plane with normal n.
    Returns (t, u) as two 3D vectors.
    """
    # Pick axis least aligned with n
    a = ti.Vector([1.0, 0.0, 0.0])
    if ti.abs(n[0]) > 0.9:
        a = ti.Vector([0.0, 1.0, 0.0])
    
    # Gram-Schmidt
    t = a - n * n.dot(a)
    t = t.normalized()
    u = n.cross(t)
    
    return t, u


@ti.func
def compute_face_area(i: ti.i32, f: ti.i32) -> ti.f32:
    """
    Compute area of face f for particle i using triangle fan.
    """
    begin = faces_begin[i, f]
    count = faces_count[i, f]
    
    if count < 3:
        return 0.0
    
    v0 = verts[i, idx[i, begin]]
    area_sum = 0.0
    
    for k in range(1, count - 1):
        vk = verts[i, idx[i, begin + k]]
        vk1 = verts[i, idx[i, begin + k + 1]]
        
        a = vk - v0
        b = vk1 - v0
        area_sum += 0.5 * a.cross(b).norm()
    
    return area_sum


@ti.kernel
def clip_all():
    """
    Clip each particle's polyhedron against all its neighbor planes.
    
    Uses Sutherland-Hodgman 3D clipping with cap face construction:
        1. For each plane, clip all faces
        2. Collect intersection points
        3. Build cap face from sorted ring
        4. Cull tiny faces
    """
    for i in range(C.N):
        if overflow[i] != 0:
            continue
        
        p_i = pos[i]
        
        # Clip against each neighbor plane
        for k in range(nbr_ct[i]):
            n = plane_n[i, k]
            b = plane_b[i, k]
            
            # Temporary storage for cap points (max 32 intersections)
            # Create array of 32 3D vectors
            cap_pts = ti.Matrix.zero(ti.f32, 32, 3)
            cap_pt_ct = 0
            
            # Temporary storage for new faces
            new_f_ct = 0
            new_i_ct = 0
            
            # For each existing face, clip it
            old_f_ct = f_ct[i]
            for f in range(old_f_ct):
                begin = faces_begin[i, f]
                count = faces_count[i, f]
                
                if count < 3:
                    continue
                
                # Clip this face against the plane
                out_poly_ct = 0
                out_poly = ti.Vector.zero(ti.i32, C.V_MAX)
                
                # Sutherland-Hodgman: walk edges
                for e in range(count):
                    v_a_idx = idx[i, begin + e]
                    v_b_idx = idx[i, begin + ((e + 1) % count)]
                    
                    v_a = verts[i, v_a_idx]
                    v_b = verts[i, v_b_idx]
                    
                    d_a = signed_dist(n, b, p_i, v_a)
                    d_b = signed_dist(n, b, p_i, v_b)
                    
                    # Case 1: both inside
                    if d_a <= EPS_PLANE and d_b <= EPS_PLANE:
                        if out_poly_ct < C.V_MAX:
                            out_poly[out_poly_ct] = v_a_idx
                            out_poly_ct += 1
                    
                    # Case 2: A inside, B outside
                    elif d_a <= EPS_PLANE and d_b > EPS_PLANE:
                        # Keep A
                        if out_poly_ct < C.V_MAX:
                            out_poly[out_poly_ct] = v_a_idx
                            out_poly_ct += 1
                        
                        # Add intersection
                        t = d_a / (d_a - d_b)
                        v_int = v_a + t * (v_b - v_a)
                        v_int_idx = dedup_and_add_vertex(i, v_int)
                        if v_int_idx >= 0:
                            if out_poly_ct < C.V_MAX:
                                out_poly[out_poly_ct] = v_int_idx
                                out_poly_ct += 1
                            # Add to cap points
                            if cap_pt_ct < 32:
                                cap_pts[cap_pt_ct, 0] = v_int[0]
                                cap_pts[cap_pt_ct, 1] = v_int[1]
                                cap_pts[cap_pt_ct, 2] = v_int[2]
                                cap_pt_ct += 1
                    
                    # Case 3: A outside, B inside
                    elif d_a > EPS_PLANE and d_b <= EPS_PLANE:
                        # Add intersection
                        t = d_a / (d_a - d_b)
                        v_int = v_a + t * (v_b - v_a)
                        v_int_idx = dedup_and_add_vertex(i, v_int)
                        if v_int_idx >= 0:
                            if out_poly_ct < C.V_MAX:
                                out_poly[out_poly_ct] = v_int_idx
                                out_poly_ct += 1
                            # Add to cap points
                            if cap_pt_ct < 32:
                                cap_pts[cap_pt_ct, 0] = v_int[0]
                                cap_pts[cap_pt_ct, 1] = v_int[1]
                                cap_pts[cap_pt_ct, 2] = v_int[2]
                                cap_pt_ct += 1
                    
                    # Case 4: both outside - skip
                
                # If clipped face has >= 3 vertices, keep it
                if out_poly_ct >= 3:
                    if new_f_ct >= C.F_MAX or new_i_ct + out_poly_ct > C.I_MAX:
                        overflow[i] = 1
                        break
                    
                    faces_begin[i, new_f_ct] = new_i_ct
                    faces_count[i, new_f_ct] = out_poly_ct
                    faces_normal[i, new_f_ct] = faces_normal[i, f]
                    
                    for p in range(out_poly_ct):
                        idx[i, new_i_ct + p] = out_poly[p]
                    
                    new_i_ct += out_poly_ct
                    new_f_ct += 1
            
            # Update counters after clipping faces
            f_ct[i] = new_f_ct
            i_ct[i] = new_i_ct
            
            # Build cap face from collected intersection points
            if cap_pt_ct >= 3 and overflow[i] == 0:
                # Compute centroid
                centroid = ti.Vector([0.0, 0.0, 0.0])
                for c in range(cap_pt_ct):
                    centroid[0] += cap_pts[c, 0]
                    centroid[1] += cap_pts[c, 1]
                    centroid[2] += cap_pts[c, 2]
                centroid /= ti.cast(cap_pt_ct, ti.f32)
                
                # Build plane frame
                t, u = plane_frame(n)
                
                # Sort cap points by angle (simple selection sort)
                cap_order = ti.Vector.zero(ti.i32, 32)
                for c in range(cap_pt_ct):
                    cap_order[c] = c
                
                for a in range(cap_pt_ct):
                    best = a
                    p_a = ti.Vector([
                        cap_pts[cap_order[a], 0] - centroid[0],
                        cap_pts[cap_order[a], 1] - centroid[1],
                        cap_pts[cap_order[a], 2] - centroid[2]
                    ])
                    ang_a = ti.atan2(p_a.dot(u), p_a.dot(t))
                    
                    for bb in range(a + 1, cap_pt_ct):
                        p_bb = ti.Vector([
                            cap_pts[cap_order[bb], 0] - centroid[0],
                            cap_pts[cap_order[bb], 1] - centroid[1],
                            cap_pts[cap_order[bb], 2] - centroid[2]
                        ])
                        ang_bb = ti.atan2(p_bb.dot(u), p_bb.dot(t))
                        if ang_bb < ang_a:
                            best = bb
                            ang_a = ang_bb
                    
                    # Swap
                    if best != a:
                        tmp = cap_order[a]
                        cap_order[a] = cap_order[best]
                        cap_order[best] = tmp
                
                # Add cap face
                if f_ct[i] < C.F_MAX and i_ct[i] + cap_pt_ct <= C.I_MAX:
                    cap_f_idx = f_ct[i]
                    faces_begin[i, cap_f_idx] = i_ct[i]
                    faces_count[i, cap_f_idx] = cap_pt_ct
                    faces_normal[i, cap_f_idx] = n
                    
                    # Add vertices in sorted order (dedup)
                    for s in range(cap_pt_ct):
                        c_idx = cap_order[s]
                        v_cap = ti.Vector([
                            cap_pts[c_idx, 0],
                            cap_pts[c_idx, 1],
                            cap_pts[c_idx, 2]
                        ])
                        v_cap_idx = dedup_and_add_vertex(i, v_cap)
                        if v_cap_idx >= 0:
                            idx[i, i_ct[i]] = v_cap_idx
                            i_ct[i] += 1
                    
                    f_ct[i] += 1
            
            if overflow[i] != 0:
                break


# =============================================================================
# Helper Functions (for clipping implementation)
# =============================================================================

@ti.func
def clip_polygon_by_plane(
    i: ti.i32,
    face_idx: ti.i32,
    n: ti.template(),
    b: ti.f32
) -> ti.i32:
    """
    Clip a single face polygon against a plane using Sutherland-Hodgman.
    
    Args:
        i: particle index
        face_idx: face index to clip
        n: plane normal (float3)
        b: plane offset
    
    Returns:
        1 if cap face created, 0 otherwise
    
    TODO: Implement Sutherland-Hodgman 3D clipping
    """
    # Get face vertices
    begin = faces_begin[i, face_idx]
    count = faces_count[i, face_idx]
    
    # Compute signed distances for all vertices
    # d_k = n·(v_k - p_i) - b  (but v_k is already relative to p_i)
    # So: d_k = n·v_k - b
    
    # TODO: Implement edge-by-edge clipping
    # - Keep vertices with d ≤ 0
    # - Insert intersection points on crossing edges
    # - Update face vertex list
    
    return 0  # placeholder


@ti.func
def build_cap_face(
    i: ti.i32,
    n: ti.template(),
    b: ti.f32,
    cap_points: ti.template(),
    cap_count: ti.i32
) -> ti.i32:
    """
    Build cap face from intersection ring.
    
    Args:
        i: particle index
        n: plane normal (float3)
        b: plane offset (unused here, but kept for consistency)
        cap_points: array of points on the cap (float3[])
        cap_count: number of cap points
    
    Returns:
        face index of new cap, or -1 if failed
    
    TODO: Implement cap face construction
    """
    # 1. Compute centroid
    # 2. Build 2D frame (t, u) on plane
    # 3. Sort points by atan2(u·(p-c), t·(p-c))
    # 4. Deduplicate (merge points within EPS_MERGE)
    # 5. Add as new face
    
    return -1  # placeholder

