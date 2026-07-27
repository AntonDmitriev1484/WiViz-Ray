import matplotlib.pyplot as plt
import numpy as np
import sionna.rt.scene as scenes
from sionna.rt import Paths
import mitsuba as mi
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


def simplify_paths(paths: Paths, scene: scenes,
             tx_idx=0, rx_idx=0,
             tx_ant_idx=0, rx_ant_idx=0):


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

    rx_paths_list = [] # Uneven array, varying number of paths per receiver?

    global_path_idx = 1
    for r_idx in range(num_rx_plot): # So we already iterate through each receiver here. We can make one paths_list per receiver

        paths_list = []

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

            paths_list.append(path_coords)

            global_path_idx += 1

        rx_paths_list.append(paths_list)

    return rx_paths_list