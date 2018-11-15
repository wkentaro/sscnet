#!/usr/bin/env python

import json
import pathlib

import matplotlib.path
import numpy as np
import scipy.io
import scipy.spatial
import trimesh

import utils
from volume_param import (
    camK,
    height_belowfloor,
    voxSize,
    voxSizeTarget,
    voxSizeCam,
    voxUnit,
)  # NOQA


def getSceneVoxSUNCG(pathToData, sceneId, floorId, roomId, extCam2World):
    # Notes: grid is Z up while the The loaded houses are Y up

    objcategory = scipy.io.loadmat(
        'suncgObjcategory.mat', struct_as_record=False, squeeze_me=True
    )['objcategory']
    # Compute voxel range in world coordinates
    voxRangeExtremesCam = np.c_[
        np.r_[-voxSizeCam[0:2] * voxUnit / 2., 0],
        np.r_[-voxSizeCam[0:2] * voxUnit / 2., 2] + voxSizeCam * voxUnit
    ]
    voxOriginCam = np.mean(voxRangeExtremesCam, 1)

    # Compute voxel grid centers in world coordinates
    voxOriginWorld = (
        extCam2World[0:3, 0:3] @ voxOriginCam +
        extCam2World[0:3, 3] -
        np.r_[
            voxSize[0] / 2. * voxUnit,
            voxSize[1] / 2. * voxUnit,
            voxSize[2] / 2. * voxUnit,
        ]
    )
    voxOriginWorld[2] = height_belowfloor
    gridPtsWorldY, gridPtsWorldX, gridPtsWorldZ = np.meshgrid(
        np.linspace(
            voxOriginWorld[1],
            (voxOriginWorld[1] + (voxSize[1] - 1) * voxUnit),
            voxSize[1],
        ),
        np.linspace(
            voxOriginWorld[0],
            (voxOriginWorld[0] + (voxSize[0] - 1) * voxUnit),
            voxSize[0],
        ),
        np.linspace(
            voxOriginWorld[2],
            (voxOriginWorld[2] + (voxSize[2] - 1) * voxUnit),
            voxSize[2],
        ),
    )
    gridPtsWorld = np.c_[
        gridPtsWorldX.flatten(),
        gridPtsWorldY.flatten(),
        gridPtsWorldZ.flatten(),
    ].T
    gridPtsLabel = np.zeros((1, gridPtsWorld.shape[1]), dtype=np.int32)

    with open(pathToData / 'house' / sceneId / 'house.json') as f:
        house = json.load(f)
    floorStruct = house['levels'][floorId]
    roomStruct = floorStruct['nodes'][roomId]

    inRoom = np.zeros(gridPtsWorldX.shape, dtype=bool)
    # find all grid in the room
    floor_file = pathToData / 'room' / sceneId / (roomStruct['modelId'] + 'f.obj')
    if floor_file.exists():
        print('Loading floor')
        floorObj = trimesh.load(str(floor_file))
        for i in range(len(floorObj.faces)):
            faceId = floorObj.faces[i]
            floorP = floorObj.vertices[faceId][:, [0, 2]]  # x, y
            path = matplotlib.path.Path(floorP)
            inRoom_i = path.contains_points(
                np.c_[gridPtsWorldX.flatten(), gridPtsWorldY.flatten()]
            ).reshape(gridPtsWorldX.shape)
            inRoom = inRoom | inRoom_i

        # find floor
        floorZ = np.mean(floorObj.vertices[:, 1])  # xzy
        gridPtsObjWorldInd = (
            inRoom.reshape(1, -1) &
            (np.abs(gridPtsWorld[2, :] - floorZ) <= (voxUnit / 2.)).reshape(1, -1)
        )
        _, classRootId, _, _, _ = utils.getobjclassSUNCG('floor', objcategory)
        gridPtsLabel[gridPtsObjWorldInd] = classRootId

    # find ceiling
    ceiling_file = (
        pathToData / 'room' / sceneId / (roomStruct['modelId'] + 'c.obj')
    )
    if ceiling_file.exists():
        print('Loading ceiling')
        ceilObj = trimesh.load(str(ceiling_file))
        ceilZ = np.mean(ceilObj.vertices[:, 1])
        gridPtsObjWorldInd = (
            inRoom.reshape(1, -1) &
            (np.abs(gridPtsWorld[2, :] - ceilZ) <= (voxUnit / 2.)).reshape(1, -1)
        )
        _, classRootId, _, _, _ = utils.getobjclassSUNCG('ceiling', objcategory)
        gridPtsLabel[gridPtsObjWorldInd] = classRootId

    # Load walls
    walls_file = pathToData / 'room' / sceneId / (roomStruct['modelId'] + 'w.obj')
    if walls_file.exists():
        print('Loading walls')
        WallObj = trimesh.load(str(walls_file))
        inWall = np.zeros(gridPtsWorldX.shape, dtype=bool)
        for wallObj in WallObj:
            assert wallObj.vertices.dtype == np.float64
            for i in range(len(wallObj.faces)):
                faceId = wallObj.faces[i]
                floorP = wallObj.vertices[faceId][:, [0, 2]]  # x, y
                path = matplotlib.path.Path(floorP)
                inWall_i = path.contains_points(
                    np.c_[gridPtsWorldX.flatten(), gridPtsWorldY.flatten()]
                ).reshape(gridPtsWorldX.shape)
                inWall = inWall | inWall_i
        gridPtsObjWorldInd = (
            inWall.reshape(1, -1) &
            (gridPtsWorld[2, :] < (ceilZ - voxUnit / 2.)) &
            (gridPtsWorld[2, :] > (floorZ + voxUnit / 2.))
        )
        _, classRootId, _, _, _ = utils.getobjclassSUNCG('wall', objcategory)
        gridPtsLabel[gridPtsObjWorldInd] = classRootId

    # Loop through each object and set voxels to class ID
    if 'nodeIndices' in roomStruct:
        print('Loading objects')
        for objId in roomStruct['nodeIndices']:
            object_struct = floorStruct['nodes'][objId]
            if 'modelId' in object_struct:
                objname = object_struct['modelId'].replace('/', '__')

                # Set segmentation class ID
                classRootName, classRootId, _, _, _ = utils.getobjclassSUNCG(
                    objname, objcategory
                )

                # Compute object bbox in world coordinates
                objBbox = np.array([
                    object_struct['bbox']['min'],
                    object_struct['bbox']['max'],
                ])[:, [0, 2, 1]].T

                # Load segmentation of object in object coordinates
                filename = (
                    pathToData /
                    'object_vox/object_vox_data' /
                    objname /
                    (objname + '.binvox')
                )
                with open(filename, 'rb') as f:
                    vox = utils.read_as_3d_array(f)
                voxels = vox.data
                scale = 1. * vox.scale / vox.dims[0]
                translate = vox.translate
                x, y, z = np.where(voxels)
                objSegPts = (np.c_[x, y, z] + 1) * scale + translate

                # Convert object to world coordinates
                extObj2World_yup = np.array(
                    object_struct['transform']
                ).reshape(4, 4).T
                objSegPts = extObj2World_yup @ np.r_[
                    objSegPts[:, [0, 2, 1]].T,    # zup -> yup
                    np.ones((1, objSegPts.shape[0])),
                ]
                objSegPts = objSegPts[[0, 2, 1]]  # yup -> zup

                # Get all grid points within the object bbox in world coordinates
                gridPtsObjWorldInd = (
                    (gridPtsWorld[0, :] >= (objBbox[0, 0] - voxUnit)) &
                    (gridPtsWorld[0, :] <= (objBbox[0, 1] + voxUnit)) &
                    (gridPtsWorld[1, :] >= (objBbox[1, 0] - voxUnit)) &
                    (gridPtsWorld[1, :] <= (objBbox[1, 1] + voxUnit)) &
                    (gridPtsWorld[2, :] >= (objBbox[2, 0] - voxUnit)) &
                    (gridPtsWorld[2, :] <= (objBbox[2, 1] + voxUnit))
                )
                gridPtsObjWorld = gridPtsWorld[:, gridPtsObjWorldInd]

                # If object is a window or door, clear voxels in object bbox
                _, wallId, _, _, _ = utils.getobjclassSUNCG('wall', objcategory)
                if classRootId == 4 or classRootId == 5:  # 4: window, 5: door
                    gridPtsObjClearInd = (
                        gridPtsObjWorldInd & (gridPtsLabel == wallId)
                    )
                    gridPtsLabel[gridPtsObjClearInd] = 0

                # Apply segmentation to grid points of object
                kdtree = scipy.spatial.cKDTree(objSegPts.T)
                # Note: dists is normalized in Python, but unnormalized in Matlab
                dists, indices = kdtree.query(gridPtsObjWorld.T)
                objOccInd = np.argwhere(dists <= (np.sqrt(3) / 2 * scale))[:, 0]
                gridPtsObjWorldLinearIdx = np.argwhere(gridPtsObjWorldInd)[:, 0]
                gridPtsLabel[:, gridPtsObjWorldLinearIdx[objOccInd]] = classRootId

    sceneVoxWorld = gridPtsLabel.copy().reshape(voxSize)

    # Remove grid points not in field of view
    extWorld2Cam = np.linalg.inv(np.vstack([extCam2World, [0, 0, 0, 1]]))
    gridPtsCam = extWorld2Cam[0:3, 0:3] @ gridPtsWorld + np.repeat(
        extWorld2Cam[0:3, 3][:, None], gridPtsWorld.shape[1], axis=1,
    )
    gridPtsPixX = gridPtsCam[0, :] * camK[0, 0] / gridPtsCam[2, :] + camK[0, 2]
    gridPtsPixY = gridPtsCam[1, :] * camK[1, 1] / gridPtsCam[2, :] + camK[1, 2]
    invalidPixInd = (
        (gridPtsPixX < 0) |
        (gridPtsPixX >= 640) |
        (gridPtsPixY < 0) |
        (gridPtsPixY >= 480)
    )
    gridPtsLabel[:, invalidPixInd] = 0

    # Remove grid points not in the room
    gridPtsLabel[:, ~inRoom.flatten() & (gridPtsLabel.flatten() == 0)] = 255

    # Change coordinate axes XYZ -> YZX
    extSwap = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    gridPtsY, gridPtsX, gridPtsZ = np.meshgrid(
        np.arange(voxSize[1]), np.arange(voxSize[0]), np.arange(voxSize[2])
    )
    gridPts = np.c_[gridPtsX.flatten(), gridPtsY.flatten(), gridPtsZ.flatten()].T
    gridPts = extSwap[0:3, 0:3] @ gridPts
    gridPts = gridPts.astype(np.int64)
    gridPtsLabel = gridPtsLabel.reshape(voxSizeTarget)
    gridPtsLabel[gridPts[0], gridPts[1], gridPts[2]] = gridPtsLabel.flatten()
    gridPtsLabel = gridPtsLabel.reshape(1, -1)

    # Save the volume
    sceneVox = gridPtsLabel.reshape(voxSizeTarget)

    return sceneVoxWorld, sceneVox


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    pathToData = pathlib.Path.home() / 'data/datasets/SUNCG/suncg'
    sceneId = '000514ade3bcc292a613a4c2755a5050'
    floorId = 0
    roomId = 0
    cameraPose = np.array([
        43.9162, 1.64774, 50.0449,
        0.0417627, -0.196116, -0.979691,
        0.00835255, 0.980581, -0.195938,
        0.55, 0.430998, 17.8815,
    ], dtype=np.float64)
    extCam2World = utils.camPose2Extrinsics(cameraPose)
    extCam2World = np.c_[
        (
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]) @
            extCam2World[0:3, 0:3] @
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        ),
        extCam2World[[0, 2, 1], 3],
    ]

    sceneVoxWorld, sceneVox = getSceneVoxSUNCG(
        pathToData=pathToData,
        sceneId=sceneId,
        floorId=floorId,
        roomId=roomId,
        extCam2World=extCam2World,
    )

    utils.show_volume(sceneVoxWorld, start_loop=False)
    utils.show_volume(sceneVox)
