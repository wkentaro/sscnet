import numpy as np


def readRLEfile(sceneVoxFilename):
    with open(sceneVoxFilename, 'rb') as f:
        voxOriginWorld = np.frombuffer(f.read(4 * 3), dtype=np.float32)
        camPoseArr = np.frombuffer(f.read(4 * 16), dtype=np.float32)
        sceneVoxRLE = np.frombuffer(f.read(), dtype=np.uint32)

    sceneVoxRLE = sceneVoxRLE.reshape(-1, 2)
    sceneVox_values = sceneVoxRLE[:, 0]
    sceneVox_repeats = sceneVoxRLE[:, 1]
    sceneVox = sceneVox_values.repeat(sceneVox_repeats)
    sceneVox = sceneVox.reshape((240, 144, 240), order='F')
    return sceneVox, camPoseArr, voxOriginWorld
