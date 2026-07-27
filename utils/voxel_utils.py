import numpy as np

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