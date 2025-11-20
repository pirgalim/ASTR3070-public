from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astroquery.gaia import Gaia

def gaia_radecs(ra_center, dec_center, radius_arcmin=10):
    """
    Query Gaia sources around a center coordinate.

    Parameters
    ----------
    ra_center, dec_center : float
        Center of field in degrees.
    radius_arcmin : float
        Search radius in arcminutes.

    Returns
    -------
    SkyCoord of Gaia stars
    """

    coord = SkyCoord(ra_center, dec_center, unit="deg")

    # ADQL cone search
    radius = (radius_arcmin / 60.0) * u.deg
    query = f"""
        SELECT ra, dec
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(u.deg).value})
        ) = 1
    """

    job = Gaia.launch_job_async(query)
    tbl = job.get_results()

    return SkyCoord(tbl["ra"], tbl["dec"], unit="deg")