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

    sceneVox, camPoseArr, voxOriginWorld = utils.readRLEfile(args.bin_file)

    print('voxOriginWorld:', voxOriginWorld)
    print('camPoseArr:', camPoseArr)

    utils.show_volume(sceneVox)


if __name__ == '__main__':
    main()
