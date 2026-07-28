import numpy as np


from utils.plot_utils import *
from utils.path_utils import *
from utils.voxel_utils import *
from utils.match_utils import *
from utils.interpolation_utils import *

def cube_vertices(center, side_length):
    """
    Generate the 8 vertices of a cube.

    Parameters
    ----------
    center : array-like of length 3
        (x, y, z) coordinates of the cube center.
    side_length : float
        Length of each cube edge in meters.

    Returns
    -------
    np.ndarray of shape (8, 3)
        Cube vertices.
    """
    center = np.asarray(center, dtype=float)
    h = side_length / 2.0

    offsets = np.array([
        [-h, -h, -h],
        [-h, -h,  h],
        [-h,  h, -h],
        [-h,  h,  h],
        [ h, -h, -h],
        [ h, -h,  h],
        [ h,  h, -h],
        [ h,  h,  h],
    ])

    return center + offsets


import numpy as np
from sionna.rt import Receiver, PathSolver

# ============================================================
# Voxel
# ============================================================

class Voxel:
    """
    A single voxel bounded by 8 receiver locations.

    receiver_ids:
        The 8 receiver indices defining this voxel.

    rx_paths:
        Simplified Sionna paths indexed by receiver id.
    """

    def __init__(
        self,
        scene,
        receiver_ids,
        rx_paths,
        grid_index,
        bounds_min,
        bounds_max
    ):

        if len(receiver_ids) != 8:
            raise ValueError(
                "Voxel requires exactly 8 receivers"
            )

        self.receiver_ids = receiver_ids
        self.grid_index = grid_index

        self.bounds_min = np.asarray(bounds_min)
        self.bounds_max = np.asarray(bounds_max)


        # Receiver positions
        self.positions = np.array([
            scene.receivers[f"rx{i}"].position
            for i in receiver_ids
        ])


        # Paths associated with these receivers
        voxel_paths = [
            rx_paths[i]
            for i in receiver_ids
        ]


        # Match equivalent paths
        match_results = match_paths_constrain_d(
            voxel_paths
        )


        # Compute interpolation coefficients
        self.path_interpolations = (
            compute_trilinear_interpolation(
                match_results,
                self.positions
            )
        )


    def contains(self, point):

        point = np.asarray(point)

        return np.all(
            point >= self.bounds_min
        ) and np.all(
            point <= self.bounds_max
        )


    def interpolate(self, point):

        """
        Evaluate all interpolated paths at point.
        """

        x, y, z = point

        basis = np.array([
            1,
            x,
            y,
            z,
            x*y,
            x*z,
            y*z,
            x*y*z
        ])


        paths = []

        for coeffs in self.path_interpolations:

            flat = basis @ coeffs

            path = flat.reshape(
                -1,
                3
            )

            paths.append(path)

        return paths



# ============================================================
# Octree
# ============================================================

class OctreeNode:

    def __init__(
        self,
        center,
        size
    ):

        self.center = np.asarray(center)
        self.size = size

        self.children = None
        self.voxel = None



    def child_index(self, point):

        idx = 0

        if point[0] >= self.center[0]:
            idx |= 1

        if point[1] >= self.center[1]:
            idx |= 2

        if point[2] >= self.center[2]:
            idx |= 4

        return idx



    def create_children(self):

        if self.children is not None:
            return


        child_size = self.size / 2

        self.children = []

        for i in range(8):

            offset = np.array([
                1 if i & 1 else -1,
                1 if i & 2 else -1,
                1 if i & 4 else -1
            ])

            child_center = (
                self.center
                +
                offset * child_size / 2
            )

            self.children.append(
                OctreeNode(
                    child_center,
                    child_size
                )
            )



    def insert(
        self,
        voxel
    ):

        self.voxel = voxel



    def query(
        self,
        point
    ):

        if self.children is None:

            return self.voxel


        idx = self.child_index(point)

        return self.children[idx].query(
            point
        )



# ============================================================
# Voxel Grid
# ============================================================

class VoxelGrid:


    def __init__(
        self,
        scene,
        bounds_min,
        bounds_max,
        spacing
    ):

        self.scene = scene

        self.bounds_min = np.asarray(
            bounds_min,
            dtype=float
        )

        self.bounds_max = np.asarray(
            bounds_max,
            dtype=float
        )

        self.spacing = spacing


        self.receiver_grid = {}

        self.voxels = {}



        # ----------------------------------------------------
        # Generate receiver grid
        # ----------------------------------------------------

        xs = np.arange(
            bounds_min[0],
            bounds_max[0] + spacing,
            spacing
        )

        ys = np.arange(
            bounds_min[1],
            bounds_max[1] + spacing,
            spacing
        )

        zs = np.arange(
            bounds_min[2],
            bounds_max[2] + spacing,
            spacing
        )


        rx_id = 0


        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                for iz, z in enumerate(zs):

                    pos = np.array(
                        [x,y,z]
                    )


                    self.scene.add(
                        Receiver(
                            name=f"rx{rx_id}",
                            position=pos
                        )
                    )


                    self.receiver_grid[
                        (ix,iy,iz)
                    ] = rx_id


                    rx_id += 1



        self.grid_shape = (
            len(xs),
            len(ys),
            len(zs)
        )


        # ----------------------------------------------------
        # Ray trace once
        # ----------------------------------------------------

        solver = PathSolver()

        paths = solver(
            self.scene,
            max_depth=5,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            diffraction=True,
            edge_diffraction=True,
            max_num_paths_per_src=100000
        )


        rx_paths = simplify_paths(
            paths,
            self.scene
        )



        # ----------------------------------------------------
        # Build octree
        # ----------------------------------------------------

        center = (
            self.bounds_min
            +
            self.bounds_max
        ) / 2


        size = np.max(
            self.bounds_max
            -
            self.bounds_min
        )


        self.root = OctreeNode(
            center,
            size
        )


        self.build_voxels(
            rx_paths,
            xs,
            ys,
            zs
        )



    def build_voxels(
        self,
        rx_paths,
        xs,
        ys,
        zs
    ):


        for ix in range(len(xs)-1):
            for iy in range(len(ys)-1):
                for iz in range(len(zs)-1):


                    corners = []

                    for dx,dy,dz in [
                        (0,0,0),
                        (1,0,0),
                        (0,1,0),
                        (1,1,0),
                        (0,0,1),
                        (1,0,1),
                        (0,1,1),
                        (1,1,1)
                    ]:

                        corners.append(
                            self.receiver_grid[
                                (
                                    ix+dx,
                                    iy+dy,
                                    iz+dz
                                )
                            ]
                        )


                    voxel_min = np.array([
                        xs[ix],
                        ys[iy],
                        zs[iz]
                    ])


                    voxel_max = (
                        voxel_min
                        +
                        self.spacing
                    )


                    voxel = Voxel(
                        self.scene,
                        corners,
                        rx_paths,
                        (ix,iy,iz),
                        voxel_min,
                        voxel_max
                    )


                    self.voxels[
                        (ix,iy,iz)
                    ] = voxel


                    # Insert into octree
                    self.insert_voxel_octree(
                        voxel
                    )



    def insert_voxel_octree(
        self,
        voxel
    ):

        center = (
            voxel.bounds_min
            +
            voxel.bounds_max
        ) / 2


        node = self.root


        # For now store directly in root.
        # Replace this with subdivision logic
        # for adaptive octrees.

        node.insert(
            voxel
        )



    def point_to_index(
        self,
        point
    ):

        """
        Convert world coordinates to integer voxel index.
        """

        relative = (
            np.asarray(point)
            -
            self.bounds_min
        ) / self.spacing


        return tuple(
            np.floor(relative)
            .astype(int)
        )



    def query(
        self,
        point
    ):

        """
        Query by floating-point world coordinate.
        """

        idx = self.point_to_index(
            point
        )


        voxel = self.voxels.get(
            idx,
            None
        )


        if voxel is None:
            return None


        return voxel.interpolate(
            point
        )