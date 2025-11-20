import numpy as np
from astropy.wcs import WCS
from astropy.modeling import models, fitting

def compute_wcs(image_points, sky_coords):
    """
    Compute a simple TAN WCS from matched image + sky coordinates.

    Parameters
    ----------
    image_points : array (N, 2)
        Pixel coordinates (x, y)
    sky_coords : SkyCoord (N)
        Matching sky coordinates

    Returns
    -------
    WCS object
    """
    x = image_points[:, 0]
    y = image_points[:, 1]

    ra = sky_coords.ra.deg
    dec = sky_coords.dec.deg

    # Fit a basic affine transform
    model = models.AffineTransformation2D()
    fitter = fitting.LinearLSQFitter()
    tform = fitter(model, np.vstack([x, y]).T, np.vstack([ra, dec]).T)

    # Build WCS object
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # crude mapping; good enough for ASTR3070 assignments
    w.wcs.crpix = [np.mean(x), np.mean(y)]
    w.wcs.crval = [np.mean(ra), np.mean(dec)]
    w.wcs.cd = np.array([[tform.matrix.value[0,0], tform.matrix.value[0,1]],
                         [tform.matrix.value[1,0], tform.matrix.value[1,1]]])

    return w