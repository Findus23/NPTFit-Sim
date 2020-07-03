###############################################################################
# ps_mc.py
###############################################################################
#
# Program that does point source Monte Carlo based off of user defined source
# count distribution, template, exposure map, and user defined point source
# function. Can save result of simulation to .npy file.
#
###############################################################################

import int_src_cnt_dist as iscd
import create_flux as cf
import make_map as mm

import numpy as np

def run(n,F,A,temp,exp,psf_r,name="map",save=False,flux_frac=np.array([1.])):
    """ Brings together serveral programs to run point source Monte Carlo by
        reading in template, source count distribution parameters, exposure 
        map, and the user defined PSF.

            :param n: numpy array of index values, highest to lowest
            :param F: numpy array of flux break values, highest to lowest
            :param A: float of log10 norm for the SCD
            :param temp: HEALPix numpy array of template
            :param exp: HEALPix numpy array of exposure map
            :param psf_r: user defined point spread function
            :param name: string for the name of output .npy file
            :param save: option to save map to .npy file
            :param flux_frac: array of flux fractions to distribute between
                different energy bins, default is 1 bin

            :returns: HEALPix format numpy array of simulated map
    """
    
    # If exposure map is a 1D array, wrap it again so it has 1 energy bin
    if len(np.shape(exp)) == 1:
        exp = np.array([exp])
    
    # Check exposure map and flux frac have same number of energy bins
    assert(len(exp) == len(flux_frac)), \
    "exposure and flux fraction must have the same number of energy bins"
    
    # Check if flux breaks in correct order
    if len(F) > 1:
        assert(F[0] > F[-1]), \
            "Flux array is in the wrong order, highest to lowest!"
    
    # Int. SCD to find mean couts, Poisson draws for # of sources in template
    num_src = iscd.run(n,F,A,temp)

    # Draws fluxes for each source from the SCD
    flux_arr = cf.run(num_src,n,F)

    # Generate simulated counts map
    map_arr = np.asarray(mm.run(num_src,flux_arr,temp,exp,psf_r,flux_frac))

    # Save the file as an .npy file
    if save:
        np.save(str(name) + ".npy",map_arr.astype(np.int32))

    print("Done simulation.")

    return map_arr.astype(np.int32)
