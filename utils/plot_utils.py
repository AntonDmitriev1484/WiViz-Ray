import matplotlib.pyplot as plt
import numpy as np
import sionna.rt.scene as scenes
from sionna.rt import Paths
import mitsuba as mi


def add_objects(scene: scenes, ax):
    all_points = []

    
    # print(scene.edit_scene_shapes)
    # Scene stores objects in a dict

    vertices = scene.__dict__['_scene_params']["merged-shapes.vertex_positions"]
    vertices = np.array(vertices)
    vertices = vertices.reshape(int(len(vertices)/3), 3)
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=2,
        color="grey"
    )

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):   # avoid duplicates/self-loops
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                color="grey",
                linewidth=1
            )


    return ax

def add_txs(txs, ax):
    ax.scatter(txs[:,0], txs[:, 1], txs[:, 2], color = "blue", s=20, label = "Tx")
    return ax

def add_rxs(rxs, ax):
    ax.scatter(rxs[:,0], rxs[:, 1], rxs[:, 2], color = "green", s=20, label = "Rx")
    return ax


def set_axes_equal(ax):
    """Make 3D plot axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([
        x_mid - max_range / 2,
        x_mid + max_range / 2
    ])
    ax.set_ylim3d([
        y_mid - max_range / 2,
        y_mid + max_range / 2
    ])
    ax.set_zlim3d([
        z_mid - max_range / 2,
        z_mid + max_range / 2
    ])



import numpy as np
def _to_numpy_xyz(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.array(x, dtype=float).reshape(3,)
def _extract_scene_positions(scene_obj_dict):
    positions = []
    names = []
    for name, obj in scene_obj_dict.items():
        pos = getattr(obj, "position", None)
        if pos is None:
            continue
        positions.append(_to_numpy_xyz(pos))
        names.append(name)
    if len(positions) == 0:
        return np.empty((0, 3)), []
    return np.vstack(positions), names

def add_rays(simple_paths, ax):

    for path in simple_paths:
        ax.plot(                
            path[:, 0],
            path[:, 1],
            path[:, 2], 
            color="red"
            )
