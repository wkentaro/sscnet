import struct

import numpy as np


def writeRLEfile(seneVoxFilename, sceneVox, camPoseArr, voxOriginWorld):
    # Compress with RLE and save to binary file (first value represents how
    # many numbers are saved to the binary file)
    # Save vox origin in world coordinates as first three values

    sceneVoxArr = sceneVox.flatten()
    sceneVoxGrad = sceneVoxArr[1:] - sceneVoxArr[0:-1]
    sceneVoxPeaks = np.argwhere(np.abs(sceneVoxGrad) > 0)

    if sceneVoxPeaks.sum() == 0:
        sceneVoxRLE = np.array([sceneVoxArr[0], sceneVoxArr.size])
    else:
        sceneVoxRLE = np.c_[
            sceneVoxArr[sceneVoxPeaks[1:]],
            (sceneVoxPeaks[1:] - sceneVoxPeaks[0:-1])
        ]
        sceneVoxRLE = np.hstack([
            sceneVoxArr[sceneVoxPeaks[0]], sceneVoxPeaks[0],
            sceneVoxRLE.flatten(),
            sceneVoxArr[sceneVoxPeaks[-1]], sceneVoxArr.size - sceneVoxPeaks[-1]
        ])

        with open(seneVoxFilename, 'wb') as f:
            voxOriginWorld = voxOriginWorld.flatten().tolist()
            f.write(struct.pack('f' * len(voxOriginWorld), *voxOriginWorld))
            camPoseArr = camPoseArr.flatten().tolist()
            f.write(struct.pack('f' * len(camPoseArr), *camPoseArr))
            sceneVoxRLE = sceneVoxRLE.flatten().tolist()
            f.write(struct.pack('I' * len(sceneVoxRLE), *sceneVoxRLE))
