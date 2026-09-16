# Right kidney model

`kidney_right.stl` is copied without modification from the reference application
`/Users/goudamasaki/Desktop/test/kidney_right.stl` supplied for this project.
Its original coordinates and millimetre units are preserved. Runtime code loads
this repository copy and has no dependency on the reference application's path.

The scan places the source mesh's bounding-box centre at the configured camera
position, then applies rotations about that centre. The closed surface is filled
with source-aligned cubes at the requested resolution before applying this pose.
