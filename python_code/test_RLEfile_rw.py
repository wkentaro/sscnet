#!/usr/bin/env python

import os
import tempfile

import utils


rle_file = '../data/SUNCGtrain_1_500/00000000_01d4be86806197794c9333540d5bf77c_fl001_rm0007_0000.bin'  # NOQA

sceneVox, camPoseArr, voxOriginWorld = utils.readRLEfile(rle_file)

rle_file2 = tempfile.mktemp() + '.bin'
utils.writeRLEfile(rle_file2, sceneVox, camPoseArr, voxOriginWorld)

sceneVox2, camPoseArr2, voxOriginWorld2 = utils.readRLEfile(rle_file2)

assert (sceneVox == sceneVox2).all()
assert (camPoseArr == camPoseArr2).all()
assert (voxOriginWorld == voxOriginWorld2).all()

os.remove(rle_file2)
