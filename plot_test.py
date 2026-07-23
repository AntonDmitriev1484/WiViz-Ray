
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

if __name__ == "__main__":
    scene = load_scene(sionna.rt.scene.simple_reflector)

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
                    position=[5, 0, 5])

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx",
                position=[-5, 0, 5])

    scene.add(rx)

    p_solver  = PathSolver()

    # Compute propagation paths
    paths = p_solver(scene=scene,
                    max_depth=5,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=False,
                    refraction=True,
                    synthetic_array=True,
                    seed=41)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    add_objects(scene, ax)
    add_rxs(rx.position, ax)
    add_txs(tx.position, ax)
    add_rays(paths, scene, ax)

    set_axes_equal(ax)
    ax.legend()

    plt.show()