################################################################################
# rej_samp.pyx
################################################################################
#
# Given a template as a numpy array, returns theta and phi position for a source
# using rejection sampeling.
#
# Update due to Florian List:
# For studies of point sources around the galactic center, setting r_ROI [deg]
# will only draw sources with r < r_ROI, for r the angle from the galactic 
# center, which can dramatically speed up the simulation timescale.
#
################################################################################

import numpy as np
import healpy as hp

cimport numpy as np
cimport cython

DTYPE = np.float

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[::1] coords(double r_ROI=np.nan):
    """ Returns an array of random theta and phi values (in radians).
    """

    # Create an empty array to hold position
    cdef double [::1] crds = np.zeros(2, dtype=DTYPE)
    # Choose random float from 0 <= x < 1 twice
    cdef double x_th_min, x_th_max, x_ph_min, x_ph_max, th_ran, ph_ran

    # If r_ROI set, only draw sources within r_ROI of the galactic center
    if not np.isnan(r_ROI):
        x_th_min = np.sin(np.radians(90.0 - r_ROI) / 2.0) ** 2.0
        x_th_max = np.sin(np.radians(90.0 + r_ROI) / 2.0) ** 2.0
        x_ph_min = np.radians(-r_ROI) / (2.0 * np.pi)
        x_ph_max = np.radians(r_ROI) / (2.0 * np.pi)

        th_ran = np.random.uniform(x_th_min, x_th_max)
        ph_ran = np.random.uniform(x_ph_min, x_ph_max)

        crds[0] = 2.0 * np.arcsin(np.sqrt(th_ran))
        crds[1] = np.mod(2.0 * np.pi * ph_ran, 2.0 * np.pi)

    # Otherwise draw coordinates uniformly on the sphere
    else:
        th_ran = np.random.random()
        ph_ran = np.random.random()
        # Calculate random theta and phi positions
        crds[0] = 2*np.arcsin(np.sqrt(th_ran))
        crds[1] = 2*np.pi*ph_ran 
    return crds

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[::1] run(double[::1] temp, double r_ROI=np.nan):
    """ Returns a source position from a give template in terms of theta and phi
        (in radians) using rejection sampling. 

            :params temp: numpy array corresponding to template
            :params r_ROI: maximum distance to draw sources from the galactic
                center
    """

    # Determine the NSIDE of the template
    cdef int NSIDE = hp.npix2nside(len(temp))
    cdef double[::1] crds = np.zeros(2)
    cdef int pos = 0
    cdef double rnd = 0.0
    # Make max pixel value in template 1.0 for rejection sampling
    temp = temp / np.max(temp)
    cdef int i = 0
    while i < 1:
        # Grab random source position in terms of theta and phi
        crds = coords(r_ROI=r_ROI)
        # Find coresponding Healpix pixel
        pos = hp.pixelfunc.ang2pix(NSIDE,crds[0],crds[1])
        # Choose random float from 0 <= x < 1
        rnd = np.random.random()
        # If the random number is less that template value, accept coords
        if rnd <= temp[pos]:
            i += 1
    # Returns coords
    return crds
