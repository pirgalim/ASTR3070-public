import numpy as np
from skimage.feature import peak_local_max

def find_peaks(data, threshold=5, min_distance=5):
    """
    Finds peaks (stars) in an image using a threshold and minimum separation.

    Parameters
    ----------
    data : 2D numpy array
        Image array.
    threshold : float
        Minimum peak intensity relative to background std.
    min_distance : int
        Minimum separation between peaks.

    Returns
    -------
    peaks : ndarray
        Nx2 array of (y, x) coordinates.
    """
    # simple background estimation
    med = np.median(data)
    std = np.std(data)

    mask = data > (med + threshold * std)

    coords = peak_local_max(
        data,
        min_distance=min_distance,
        threshold_abs=med + threshold * std,
        exclude_border=False
    )

    return coords