#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This is a set of routines to read in the eigenbasis of a Super-Colours eigensystem
and the photometry data from the SIMBA simulation and its required adaptation to the
eigensystem used.

Author: Curro Rodriguez Montero
Email: s1650043@ed.ac.uk
'''
## IMPORT LIBRARIES
from astropy.io import fits
import numpy as np
import h5py
##

def read_eigensystem(evecfile, verbose=True):
    '''
    This routine reads in the eigenbasis of super colours in .fits format and
    spits out the wavelength, eigenvectors, variance  and mean spectrum.
    '''
    fits_path = str(evecfile)
    hdul = fits.open(fits_path) # Read in .fits file
    data = hdul[1].data # Saving data as numpy array
    hdul.close() # Once we're done with the .fits file, close

    # Save data in numpy arrays
    wave = data['WAVE_REST_SUPER'][0]
    spec = data['EVECS'][0]
    mean = data['MEANARR'][0]
    var = data['VARIANCE'][0]
    ind = data['IND_WAVE'][0]
    minz = data['MINZ'][0]
    maxz = data['MAXZ'][0]
    dz = data['DZ'][0]
    filternames = data['FILTERNAMES_SUPER'][0]

    # Print some info about the eigensystem
    if verbose:
        print(str(len(spec))+' eigenvectors extracted')
        print('Minimum redshift of the survey used: '+str(minz))
        print('Maximum redshift of the survey used: '+str(minz))
    return wave,spec,mean,var,ind,minz,maxz,dz,filternames

def mag_to_jansky(mag_AB):
    '''
    From absolute magnitude array, returns the equivalent array of the flux in units
    of Jansky (1Jy = 10-23 erg s-1 Hz-1 cm-2).
    '''
    f_nu_ergs = 10**(-0.4*(mag_AB + 48.6))
    f_nu = f_nu_ergs*(1**23)

    return f_nu

def data_from_simba(ph_file, magcols, mag_lim, ind_select):
    '''
    This routine reads in the selected apparent magnitudes from the SIMBA loser
    files in order to:
    - Get errors in the magnitudes
    - Apply magnitude cut
    - Converts magnitudes into fluxes
    - Add error floors
    - Obtain redshift array
    - Obtain K-mag array
    '''
    f = h5py.File(ph_file,'r') # Read in .hdf5 file with photometry catalogue
    header = f['HEADER_INFO']
    redshift = header[0].split()[2] # Get redshift of snapshot
    Lapp = []
    for i in range(len(magcols)):
        imag = int(magcols[i])
        Lapp.append(f['appmag_%d'%imag]) # Save mags for the selected filters
    Lapp = np.asarray(Lapp)  # Apparent magnitudes of galaxies in each desired band
    print(Lapp)
    print(np.where(Lapp<mag_lim))
    Lapp = Lapp[np.where(Lapp<mag_lim)] # Apply magnitude limit given by mag_lim
    print(Lapp)
    Lapp_err = np.full((len(Lapp)),0.01) # Create array with magnitude errors

    flux = mag_to_jansky(Lapp)
    flux_err = flux - mag_to_jansky(Lapp + Lapp_err)

    # Adding error floors due to systematic errors in filters
    irac_bands = [17,18]
    flux_err = flux_err + 0.05*flux # 0.05 for all
    flux_err[irac_bands[0]:irac_bands[1]] = flux_err[irac_bands[0]:irac_bands[1]] + 0.2*flux[irac_bands[0]:irac_bands[1]] # 0.2 for the IRAC bands

    # Create array with redshifts
    z = np.full(len(Lapp[0]), redshift)

    # Get magnitude of selection filter
    Kmag = Lapp[ind_select]

    return flux, flux_err, z, Kmag
