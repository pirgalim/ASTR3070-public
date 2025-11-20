import numpy as np

def sparsify(coords, step=5):
    """
    Downsample a list of coordinates for faster matching.

    Parameters
    ----------
    coords : (N, 2) array
        x, y coordinates
    step : int
        Keep every 'step'-th point

    Returns
    -------
    (M, 2) array
    """
    return coords[::step]