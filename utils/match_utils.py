

from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
import numpy as np


from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import math


def create_batches(rx_paths):
    """
    Returns A such that

        A[path_length][receiver]

    is a list of all paths having exactly `path_length` vertices
    observed by that receiver.
    """

    nrx = len(rx_paths)

    # Largest path length present
    max_path_length = max(
        len(path)
        for paths in rx_paths
        for path in paths
    )

    # Index 0 is unused so indexing matches path length
    A = [[[] for _ in range(nrx)] for _ in range(max_path_length + 1)]

    for rx, paths in enumerate(rx_paths):
        for path in paths:
            A[len(path)][rx].append(path)

    return A

def match_paths_constrain_n(rx_paths, plot_pca=False):

    A = create_batches(rx_paths)

    eps = [] # create one distance threshold per path dimension clustering

    # Its claiming there are 0 paths of length 1
    # Oh I see, technically a LoS path is of length 2 as it includes the Tx and Rx vertex.
    for dim in range(2, len(A)):
        print(f"{dim=}")
        nrx = len(A[dim])
        npaths_per_rx = [len(A[dim][i]) for i in range(nrx)]
        k_estimate = np.mean( np.array( npaths_per_rx ))
        print(f" {k_estimate=}")

        # ex. How many 3 dimensional paths does each receiver have?
        # If each receiver has 2 3D paths, we would expect 2 clusters, of size 8.

        # Now run constrained kmeans on each dimension

        flattened = []
        rx_labels = []
        for i in range(0,nrx): # For each receiver
            npaths = len(A[dim][i])
            for j in range(0, npaths):
                flat_path = np.ravel(np.array(A[dim][i][j]))
                flattened.append(flat_path)
                rx_labels.append(i)
        flattened = np.array(flattened)

        n_clusters = math.ceil(k_estimate)
        size_min = 8
        if size_min * n_clusters > flattened.shape[0]: size_min = 7
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min = size_min,
            size_max = 8,
            random_state = 0
        )



        # Current problem is that there are 50 paths,
        # But k_estimate = 6.25, not all receivers have the same number of paths of this dimension
        # Some have 5 paths, others have 7

        # size_min=8 * n_clusters < 50

        # So this is where we must vary size_max to get the clustering running
        # Then prune clusters that don't belong.

        idxs = clf.fit_predict(flattened, y=flattened)

        for i in range(n_clusters):
            cluster_elements = flattened[idxs == i]
            cluster_center = clf.cluster_centers_[i]

            distances = np.linalg.norm(cluster_elements - cluster_center, axis=1)

            eps.append(np.mean(distances))

        # print(clf.cluster_centers_)
        # flattened: shape (num_paths, num_features)
        # idxs: cluster labels returned by fit_predict()


        if plot_pca:
            # Project to 2D
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(flattened)

            # Plot
            plt.figure(figsize=(8, 8))

            unique_clusters = np.unique(idxs)

            for cluster in unique_clusters:
                mask = idxs == cluster
                plt.scatter(
                    X2[mask, 0],
                    X2[mask, 1],
                    s=50,
                    label=f"Cluster {cluster}"
                )

            # Plot projected cluster centers
            centers2 = pca.transform(clf.cluster_centers_)
            plt.scatter(
                centers2[:, 0],
                centers2[:, 1],
                marker="x",
                s=200,
                linewidths=3,
                color="black",
                label="Centers"
            )

            # Label each point with its original index
            for i, (x, y) in enumerate(X2):
                plt.text(x, y, str(i), fontsize=8)

            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title(f"dim={dim} paths Constrained K-Means Clusters")
            plt.legend()
            plt.axis("equal")
            plt.show()

    print(f" Avg cluster center distances for each dimension {eps=}")
    return eps

def match_paths_constrain_d(rx_paths, plot_pca=False):

    A = create_batches(rx_paths)

    results = []

    # Its claiming there are 0 paths of length 1
    # Oh I see, technically a LoS path is of length 2 as it includes the Tx and Rx vertex.
    for dim in range(2, len(A)):
        nrx = len(A[dim])
        npaths_per_rx = [len(A[dim][i]) for i in range(nrx)]
        k_estimate = np.mean( np.array( npaths_per_rx ))


        rx_labels = [] # For this path index in flattened, what is the corresponding rx index?
        flattened = []
        for i in range(0,nrx): # For each receiver
            npaths = len(A[dim][i])
            for j in range(0, npaths):
                flat_path = np.ravel(np.array(A[dim][i][j]))
                flattened.append(flat_path)
                rx_labels.append(i)
        flattened = np.array(flattened)


        eps =  3 # TODO: Select this dynamically!

        clustering = DBSCAN(
            eps=eps,
            min_samples=8
        )

        cluster_labels = clustering.fit_predict(flattened) # For this path index in flattened, what cluster is it part of?

        results.append( (flattened, cluster_labels, rx_labels) )


        if plot_pca:
            # Project to 2D
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(flattened)


            # Plot
            plt.figure(figsize=(8, 8))

            unique_clusters = np.unique(cluster_labels)

            for cluster in unique_clusters:

                mask = cluster_labels == cluster

                if cluster == -1:
                    # DBSCAN noise
                    plt.scatter(
                        X2[mask, 0],
                        X2[mask, 1],
                        s=50,
                        marker="x",
                        label="Noise"
                    )
                else:
                    plt.scatter(
                        X2[mask, 0],
                        X2[mask, 1],
                        s=50,
                        label=f"Cluster {cluster}"
                    )

            # Label each point with original index
            for i, (x, y) in enumerate(X2):
                plt.text(
                    x,
                    y,
                    str(i),
                    fontsize=8
                )

            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title(f"dim={dim} paths DBSCAN Clusters (eps={eps})")
            plt.legend()
            plt.axis("equal")
            plt.show()

    return results

from sklearn.neighbors import NearestNeighbors

def eval_match(interpolated_paths, gt_paths):

    # Batch just by path dimension

    # Largest path length present
    max_path_length = max(
        len(path)
        for path in gt_paths
    )

    # Index 0 is unused so indexing matches path length
    A = [[] for _ in range(max_path_length + 1)]
    GT = [[] for _ in range(max_path_length + 1)]

    for path in interpolated_paths:
        A[len(path)].append(path)

    for path in gt_paths:
        GT[len(path)].append(path)


    reflection_point_error = []

    for dim in range(2, max_path_length):

        A_flat = []
        npaths = len(A[dim])
        for j in range(0, npaths):
            flat_path = np.ravel(np.array(A[dim][j]))
            A_flat.append(flat_path)
        A_flat = np.array(A_flat)

        GT_flat = []
        npaths = len(GT[dim])
        for j in range(0, npaths):
            flat_path = np.ravel(np.array(GT[dim][j]))
            GT_flat.append(flat_path)
        GT_flat = np.array(GT_flat)

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(GT_flat)

        # Query with the paths to be matched
        distances, indices = knn.kneighbors(A_flat)

        # Ensure each GT path is matched at most once
        matched_gt = indices[:, 0]

        unique_gt, counts = np.unique(matched_gt, return_counts=True)
        duplicates = unique_gt[counts > 1]

        if len(duplicates) > 0:
            duplicate_info = []
            for gt_idx in duplicates:
                a_paths = np.where(matched_gt == gt_idx)[0]
                duplicate_info.append(
                    f"GT path {gt_idx} matched by estimated paths {a_paths.tolist()}"
                )

            raise RuntimeError(
                "Nearest-neighbor matching is not one-to-one.\n"
                + "\n".join(duplicate_info)
            )


        for i, (dist, idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
            # print(f"A path {i} -> GT path {idx} (distance={dist})")

            A_path = A_flat[i].reshape(-1, 3)
            GT_path = GT_flat[idx].reshape(-1, 3)

            # Compare 3D distance between all reflection points
            for j in range(0,dim):
                reflection_point_error.append(np.linalg.norm(A_path[j,:]-GT_path[j,:]))

    print(f" Average reflection point error is {np.mean(np.array(reflection_point_error))}m")

    return reflection_point_error