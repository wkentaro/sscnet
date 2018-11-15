import labelme
import numpy as np
import trimesh


def show_volume(volume, **kwargs):
    keep = ~np.isin(volume, [0, 255])
    points = np.argwhere(keep)
    labels = volume[keep]

    colormap = labelme.utils.label_colormap()
    colors = colormap[labels]

    pc = trimesh.PointCloud(vertices=points, color=colors)
    pc.show(**kwargs)
