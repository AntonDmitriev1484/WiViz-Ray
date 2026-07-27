
# Import or install Sionna
import sionna.rt
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

no_preview = False # Toggle to False to use the preview widget

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies

print(f"Mitsuba variant: {mi.variant()}")

from utils.plot_utils import *
from utils.path_utils import *
from utils.voxel_utils import *
from utils.match_utils import *

if __name__ == "__main__":
    scene = load_scene(sionna.rt.scene.simple_wedge)

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")

    # Create transmitter
    tx = Transmitter(name="tx",
                    position=[30, -20, 0])

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a voxel of receivers
    # spaced 1 meter apart
    offsets = cube_vertices([10,-35, 0], 1)
    for i in range(offsets.shape[0]):
        rx = Receiver(name=f"rx{i}",
                    position=offsets[i])
        scene.add(rx)

    p_solver  = PathSolver()

    full_paths = p_solver(
        scene,
        max_depth=5,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=True,
        edge_diffraction=True,
        max_num_paths_per_src=100000,
        seed=41,
    )

    
    rx_paths = simplify_paths(full_paths, scene)
    # match_paths_constrain_n(rx_paths)
    results = match_paths_constrain_d(rx_paths) 
    # Problem is that currently once we compute results we can't index back to the original format
    # So no visual debugging besides kmeans / PCA right now.



    # for i, paths in enumerate(rx_paths):


    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     add_objects(scene, ax)
    #     add_rxs( scene.receivers[f"rx{i}"].position, ax)
    #     add_txs(tx.position, ax)
    #     add_rays(paths, ax)
    #     set_axes_equal(ax)
    #     ax.legend()
    #     ax.set_xlabel("X (m)")
    #     ax.set_ylabel("Y (m)")
    #     ax.set_zlabel("Z (m)")
    #     ax.set_title (f" Rx {i} paths")
    #     plt.show()