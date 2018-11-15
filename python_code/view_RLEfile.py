#!/usr/bin/env python

import argparse

import numpy as np

import utils


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('bin_file')
    args = parser.parse_args()

    with open(args.bin_file, 'rb') as f:
        voxOriginWorld = np.frombuffer(f.read(4 * 3), dtype=np.float32)
        camPoseArr = np.frombuffer(f.read(4 * 16), dtype=np.float32)

        print('voxOriginWorld:', voxOriginWorld)
        print('camPoseArr:', camPoseArr)

        sceneVoxRLE = np.frombuffer(f.read(), dtype=np.uint32)
        sceneVoxRLE = sceneVoxRLE.reshape(-1, 2)
        sceneVox_values = sceneVoxRLE[:, 0]
        sceneVox_repeats = sceneVoxRLE[:, 1]
        sceneVox = sceneVox_values.repeat(sceneVox_repeats)
        sceneVox = sceneVox.reshape((240, 144, 240), order='F')

    utils.show_volume(sceneVox)


if __name__ == '__main__':
    main()
