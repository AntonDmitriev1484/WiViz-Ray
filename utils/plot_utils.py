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

    for i in range(vertices.shape[0]):
        for j in range(vertices.shape[1]):
            if i != j:
                ax.plot(vertices[i,:], vertices[j,:], color="grey")

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

def add_rays(paths: Paths, scene: scenes, ax, 
             tx_idx=0, rx_idx=0,
             tx_ant_idx=0, rx_ant_idx=0):


    # print(scene.receivers)

    tx_positions, _ = _extract_scene_positions(scene.transmitters)
    rx_positions_scene, _ = _extract_scene_positions(scene.receivers)
    INVALID_OBJECT_ID_UINT32 = np.uint32(4294967295)

    vertices = np.array(paths.vertices)
    objects = np.array(paths.objects)
    valid = np.array(paths.valid)

    if vertices.ndim != 5:
        raise ValueError(
            f"Expected paths.vertices to have 5 dims [depth, rx, rx_ant, tx, tx_ant, path, 3], "
            f"but got shape {vertices.shape}"
        )
    if objects.ndim != 4:
        raise ValueError(
            f"Expected paths.objects to have 4 dims [depth, rx, rx_ant, path], "
            f"but got shape {objects.shape}"
        )
    if valid.ndim != 3:
        raise ValueError(
            f"Expected paths.valid to have 3 dims [rx, rx_ant, path], "
            f"but got shape {valid.shape}"
        )

    depth_dim, num_rx, num_rx_ant, num_paths, xyz_dim = vertices.shape
    if xyz_dim != 3:
        raise ValueError(f"Last dimension of vertices must be 3, got {xyz_dim}")

    rx_ant_idx = 0
    if rx_ant_idx >= num_rx_ant:
        raise ValueError(f"Requested rx_ant_idx=0 but num_rx_ant={num_rx_ant}")

    
    num_rx_plot = min(num_rx, len(rx_positions_scene))

    global_path_idx = 1
    for r_idx in range(num_rx_plot):
        for p_idx in range(num_paths):
            if not bool(valid[r_idx, rx_ant_idx, p_idx]):
                continue

            path_coords = [tx_positions[tx_idx]]

            for d_idx in range(depth_dim):
                oid = objects[d_idx, r_idx, rx_ant_idx, p_idx]
                if np.issubdtype(type(oid), np.unsignedinteger):
                    if oid == INVALID_OBJECT_ID_UINT32:
                        continue
                else:
                    if int(oid) == -1:
                        continue

                vertex = np.array(vertices[d_idx, r_idx, rx_ant_idx, p_idx], dtype=float)
                path_coords.append(vertex)

            path_coords.append(rx_positions_scene[r_idx])
            path_coords = np.array(path_coords, dtype=float)

            cleaned = [path_coords[0]]
            for q in path_coords[1:]:
                if not np.allclose(q, cleaned[-1], atol=1e-9):
                    cleaned.append(q)
            path_coords = np.array(cleaned, dtype=float)

            if len(path_coords) < 2:
                continue

            print(path_coords)

            # ax.scatter(
            #     path_coords[:, 0],
            #     path_coords[:, 1],
            #     path_coords[:, 2],
            #     label=f"{global_path_idx}",
            #     color = "red"
            # )

            ax.plot(                
                path_coords[:, 0],
                path_coords[:, 1],
                path_coords[:, 2], 
                color="red"
                )

            # TODO: Plot connection with lines
            global_path_idx += 1