import numpy as np


voxSizeCam = np.array([240, 240, 240])
voxSize = np.array([240, 240, 144])
voxSizeTarget = np.array([240, 144, 240])
voxUnit = 0.02
voxMargin = 5
camK = np.array([
    [518.8579, 0, 320],
    [0, 518.8579, 240],
    [0, 0, 1],
])
height_belowfloor = -0.05
im_w = 640
im_h = 480
