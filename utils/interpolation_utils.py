import math
import matplotlib.pyplot as plt
import numpy as np


def compute_trilinear_interpolation(match_results, scene):

    # ------------------------------------------------------------------
    # Build the interpolation matrix X from the receiver positions.
    # ------------------------------------------------------------------

    rx_positions = np.array([
        np.asarray(scene.receivers[f"rx{i}"].position, dtype=float)
        for i in range(8)
    ])

    X = np.column_stack([
        np.ones(8),
        rx_positions[:, 0],
        rx_positions[:, 1],
        rx_positions[:, 2],
        rx_positions[:, 0] * rx_positions[:, 1],
        rx_positions[:, 0] * rx_positions[:, 2],
        rx_positions[:, 1] * rx_positions[:, 2],
        rx_positions[:, 0] * rx_positions[:, 1] * rx_positions[:, 2],
    ])

    Xinv = np.linalg.inv(X)

    path_interpolations = []

    for dim, results in enumerate(match_results):

        flattened, cluster_labels, rx_labels = results

        for label in np.unique(cluster_labels):

            # Ignore DBSCAN noise
            if label == -1:
                continue

            mask = cluster_labels == label

            c = flattened[mask]
            rx = np.asarray(rx_labels)[mask]

            # Require exactly one path from each receiver
            if len(np.unique(rx)) != 8:
                print("Cluster does not contain one path from each receiver.")
                continue

            # ----------------------------------------------------------
            # Reorder the paths so row i corresponds to receiver i.
            # ----------------------------------------------------------

            c_ordered = np.empty_like(c)

            for i, rx_id in enumerate(rx):
                c_ordered[rx_id] = c[i]

            # ----------------------------------------------------------
            # Solve for interpolation coefficients.
            #
            # Each column of c_ordered is interpolated independently.
            # ----------------------------------------------------------

            a = Xinv @ c_ordered

            path_interpolations.append(a)

    # We should end up with path_interpolation being of
    # length = sum (N clusters) over each dim

    print(path_interpolations)
    
    return path_interpolations


# def compute_trilinear_interpolation(match_results, scene):
    # https://en.wikipedia.org/wiki/Trilinear_interpolation

    # Need to tie every path, back to its rx location, so that we can solve the trilinear interpolation equation.
    # Can't get c00 c01, etc, without those rx locations.

    
    # Query the position of the associated receiver.
    #scene.receivers[f"rx{i}"].position

    X = [] # Pre-compute 8x8 rx position matrix.
    path_interpolations = []

    for dim, results in enumerate(match_results):
        flattened, cluster_labels, rx_labels = results

        print(f"{dim=}")

        unique_clusters =  list(np.unique(cluster_labels))
        print(unique_clusters)
        for label in unique_clusters:
  
            # -1 is noise label
            if label != -1: 
                mask = cluster_labels == label 
                c = flattened[mask] # Get all paths in this cluster
                rx = np.array(rx_labels)[mask] # Get all rx for this cluster

                print(c)
                print(rx)

                # Need to make sure that each rx is unique. Should never get two paths from the same rx in the cluster.

                if np.unique(rx).shape[0] != 8: 
                    print("Paths in cluster don't come from 8 distinct Rx!!!")
                    continue

                # Compute a = X^-1 @ c


                path_interpolations.append(a)

    return None