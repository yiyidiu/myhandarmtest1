# Finger collision geometry

The physical-grasp profile uses these meshes for Gazebo contact and fast
distance queries. Finger/palm self-collision decisions additionally use the
original visual meshes as described below. Visuals, inertias and the original
1.3963 rad flexion limits are unchanged.

The old cylinder/sphere proxies reported `f3link2/handbase_link` collision
at the recorded 1.23865 rad flexion. Checking the original two visual meshes
at that same pose instead gave 6.353 mm separation. A single convex hull of
a proximal link also fills its side recess and creates a false overlap near
full flexion, so the proximal links use three convex sections.

Source files: `handarmtest1/meshes/myhand/{link}.STL`, in metres.
Generation used trimesh 4.4.9 with SciPy's convex hull implementation:

```python
axis = np.array([0.044995, 0.053623, 0.0])
axis /= np.linalg.norm(axis)
parts = [
    mesh.slice_plane(axis * 0.017, -axis),
    mesh.slice_plane(axis * 0.017, axis).slice_plane(axis * 0.035, -axis),
    mesh.slice_plane(axis * 0.035, axis),
]
for i, part in enumerate(parts):
    part.convex_hull.export(f"{link}_collision_part{i}.stl")
```

This applies independently to `f1link2`, `f2link1`, and `f3link2`. The cut
stations separate the 17 mm root hub and the 35 mm link midpoint. Distal
links `f1link3`, `f2link2`, and `f3link3` use `mesh.convex_hull` as `part0`.
Each original triangle is partitioned by the planes and contained by its
section's hull; the method adds conservative volume rather than shrinking
the source surface. Original vertices and face centres were also checked
against the exported hull union, with none outside by more than 1 micrometre.

These meshes are approximations, not certified hardware clearance models.
The measured-state and swept self-collision checks still include finger/palm
pairs. The renderer uses the existing hulls for the two spread/root links.
Regeneration dependencies are not needed to run teleoperation; the generated
STLs are shipped with the source.

## Palm contact correction after the recorded box approach

**2026-09-19 update:** the physical-grasp profile now uses the original
`handbase_link.STL` for the palm in both Gazebo and MoveIt. The rounded 2 mm
mesh below is retained only as historical data. It falsely intersected
`link_4` at the 20:50 recording's stop while the original surfaces retained
1.197 mm clearance. Using the original palm in the shared robot description
keeps binary contact checks, QP distance constraints, trajectory prediction,
and simulator contact on the same palm surface. Exact palm/finger checks
remain necessary because the finger collision meshes are still convex
approximations. Arm geometry, contact pairs, and joint limits are retained.

The next live stop occurred with an OPEN command and only 0.17723 rad of
third-finger flexion. The spread joint had been deflected by object contact
from its 0.18 rad target to 0.01356 rad. The old 8 mm palm grid reported
intersection; the original palm and finger meshes still had 2.416 mm gap.
The captured 31.345 s arm-reference trajectory reproduced this stop.

`handbase_link_collision_2mm.stl` is generated from the original palm STL by
rounding each vertex to a 0.002 m grid, merging equal vertices, and removing
degenerate/duplicate faces. Its maximum source-vertex displacement is
1.723 mm; this is a physics/distance approximation, not an exact contact
surface. The physical profile also uses the original link_5 wrist mesh;
its old simplified mesh falsely intersected the correctly shaped palm.

The self-collision guard caches original palm/finger meshes. Conservative
bounding boxes select possible pairs, then exact triangle intersection
checks decide measured and predicted finger/palm collisions. These checks
do not depend on the rounded palm reporting a collision first. Consequently
rounding cannot by itself latch this fault or hide an original-mesh contact.
True adjacent joint bodies keep their existing assembly exceptions; other
self-collision pairs retain their checks. Original-mesh loading failure
stops initialization rather than silently disabling protection.

Exact palm contacts are evaluated in the palm frame using tight transformed
bounding boxes. Safe results may be reused only for exactly equal numerical
values of all eight hand-joint angles (ordinary floating-point equality,
without rounding or tolerance). Arm-only motion cannot change these relative
surfaces; all other self-collision pairs are checked again on every call.
