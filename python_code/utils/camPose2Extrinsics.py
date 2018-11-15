import numpy as np


def camPose2Extrinsics(cameraPose):
    # cameraPose : &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz
    # extrinsics : camera to world
    tv = cameraPose[3:6]
    uv = cameraPose[6:9]
    rv = np.cross(tv, uv)

    extrinsics = np.c_[
        rv,
        -cameraPose[6:9],
        cameraPose[3:6],
        cameraPose[0:3],
    ]
    return extrinsics
