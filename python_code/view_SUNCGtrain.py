#!/usr/bin/env python

import os
import pathlib
import tempfile

import labelme
import numpy as np
import trimesh

import pyglet

import mvtk


data_dir = pathlib.Path('../data/SUNCGtrain_1_500')

for bin_file in sorted(data_dir.glob('*.bin')):
    print(bin_file)

    basename = bin_file.name.split('.')[0]
    label_file = bin_file.parent / (basename + '_label.png')
    depth_file = bin_file.parent / (basename + '.png')

    lbl = mvtk.io.imread(label_file)
    colormap = labelme.utils.label_colormap()
    viz = (colormap[lbl] * 255).astype(np.uint8)

    depth = mvtk.io.imread(depth_file)
    depth = depth.astype(np.float32)
    depth[depth == 0] = np.nan
    viz2 = mvtk.image.colorize_depth(depth)

    tmp_file = tempfile.mktemp() + '.png'
    mvtk.io.imsave(tmp_file, np.hstack([viz, viz2]))
    image = pyglet.image.load(tmp_file)
    os.remove(tmp_file)

    window = pyglet.window.Window(width=image.width, height=image.height)
    sprite = pyglet.sprite.Sprite(image)

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    @window.event()
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.close()

    with open(bin_file, 'rb') as f:
        voxOriginWorld = np.frombuffer(f.read(4 * 3), dtype=np.float32)
        camPoseArr = np.frombuffer(f.read(4 * 16), dtype=np.float32)
        sceneVoxRLE = np.frombuffer(f.read(), dtype=np.uint32)
        sceneVoxRLE = sceneVoxRLE.reshape(-1, 2)
        sceneVox_values = sceneVoxRLE[:, 0]
        sceneVox_repeats = sceneVoxRLE[:, 1]
        voxSize = (240, 144, 240)
        sceneVox = sceneVox_values.repeat(sceneVox_repeats).reshape(voxSize)

    keep = ~np.isin(sceneVox, [0, 255])
    points = np.argwhere(keep)
    labels = sceneVox[keep]
    print(np.unique(labels))

    colormap = labelme.utils.label_colormap()
    colors = colormap[labels]

    pc = trimesh.PointCloud(vertices=points, color=colors)
    pc.show()
