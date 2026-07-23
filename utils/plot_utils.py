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
    ax.scatter(txs[:,0], txs[:, 1], txs[:, 2], color = "red", s=10, label = "Tx")
    return ax

def add_rxs(rxs, ax):
    ax.scatter(rxs[:,0], rxs[:, 1], rxs[:, 2], color = "green", s=10, label = "Rx")
    return ax

# def add_rays(paths:Paths, ax):

#     vertices = paths.vertices
#     if vertices.ndim != 7:
#         raise ValueError(
#             f"Expected paths.vertices to have 7 dims max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 3, "
#             f"but got shape {vertices.shape}"
#         )

#     max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, ndim = vertices.shape

# # Convert to numpy array
#     vertices = np.array(paths.vertices)

#     print("Ray array shape:", vertices.shape)
    

#     for ray_id in range(num_paths):
#         ray = vertices[:,:,:,:, :, ray_id,:]  # shape (N_vertices, 3)

#         ax.plot(
#             ray[:, 0],
#             ray[:, 1],
#             ray[:, 2],
#             color="blue",
#             linewidth=2,
#             label=f"Ray {ray_id}"
#         )

#         # Mark vertices
#         ax.scatter(
#             ray[:,0],
#             ray[:,1],
#             ray[:,2],
#             color="blue",
#             s=10
#         )

#         # Label ray start
#         ax.text(
#             ray[0,0],
#             ray[0,1],
#             ray[0,2],
#             f"{ray_id}",
#             fontsize=8
#         )


#     return ax


import numpy as np

def add_rays(paths, scene, ax, 
             tx_idx=0, rx_idx=0,
             tx_ant_idx=0, rx_ant_idx=0):

    # I definitely don't think this plotting code is correct....
    
    vertices = np.array(paths.vertices)
    print(vertices)
    valid = np.array(paths.valid)

    if vertices.ndim != 7:
        raise ValueError(
            f"Expected paths.vertices shape "
            f"[depth, rx, rx_ant, tx, tx_ant, path, 3], got {vertices.shape}"
        )

    max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, xyz_dim = vertices.shape

    if xyz_dim != 3:
        raise ValueError("Last dimension of vertices must be 3")

    # Extract scene TX/RX positions
    tx_list = list(scene.transmitters.values())
    rx_list = list(scene.receivers.values())

    tx_position = np.array(tx_list[tx_idx].position, dtype=float).reshape(3)
    rx_position = np.array(rx_list[rx_idx].position, dtype=float).reshape(3)

    ray_id = 0

    for p_idx in range(num_paths):

        # Skip invalid path
        if not valid[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, p_idx]:
            continue

        path_coords = [tx_position]

        # Add reflection vertices
        for d_idx in range(max_depth):
            vertex = np.array(
                vertices[
                    d_idx,
                    rx_idx,
                    rx_ant_idx,
                    tx_idx,
                    tx_ant_idx,
                    p_idx
                ],
                dtype=float
            ).reshape(3)

            path_coords.append(vertex)

        # Add receiver
        path_coords.append(rx_position)

        path_coords = np.array(path_coords, dtype=float)

        # Remove repeated points
        cleaned = [path_coords[0]]
        for p in path_coords[1:]:
            if not np.allclose(p, cleaned[-1], atol=1e-9):
                cleaned.append(p)

        path_coords = np.array(cleaned)

        if len(path_coords) < 2:
            continue

        # Plot ray
        ax.plot(
            path_coords[:,0],
            path_coords[:,1],
            path_coords[:,2],
            linewidth=2,
            label=f"Ray {ray_id}"
        )

        # Plot vertices
        ax.scatter(
            path_coords[:,0],
            path_coords[:,1],
            path_coords[:,2],
            s=15
        )

        # Label ray
        ax.text(
            path_coords[0,0],
            path_coords[0,1],
            path_coords[0,2],
            f"{ray_id}",
            fontsize=8
        )

        ray_id += 1

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