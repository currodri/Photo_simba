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
## CONSTANTS
c_in_AA = 2.99792**18
sol_lum_in_erg = 3.839**33
pc_in_m = 3.08567758**16
Mpc_in_cm = 3.08567758**23
##

def read_eigensystem(evecfile, filterfile, verbose=True):
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

    # Read in filter effective wavelengths into array
    f = open(filterfile).readlines()
    ll_eff = np.zeros(len(f))
    for i in range(0, len(f)):
        ll_eff[i] = float(f[i].split()[1])
    return wave,spec,mean,var,ind,minz,maxz,dz,filternames,ll_eff

def mag_to_jansky(mag_AB):
    '''
    From absolute magnitude array, returns the equivalent array of the flux in units
    of Jansky (1Jy = 10-23 erg s-1 Hz-1 cm-2).
    '''
    f_nu_ergs = 10**(-0.4*(mag_AB + 48.6))
    f_nu = f_nu_ergs*(1**23)

    return f_nu

def data_from_simba(ph_file, n_bands, mag_lim, ind_filt, ind_select):
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
    caesar_id = f['CAESAR_ID'][:]
    header = f['HEADER_INFO']
    redshift = float(header[0].split()[2]) # Get redshift of snapshot
    Lapp_old = np.zeros((len(caesar_id),n_bands))
    ind = [1,3,4,5,6,7,8,9,10,11,12]
    # Apparent magnitudes of galaxies in each desired band
    for (i,i_filt) in zip(ind, ind_filt):
        Lapp_old[:,i_filt] = f['appmag_%d'%i] # Save mags for the selected filters
    # Apply magnitude limit given by mag_lim
    Lapp = []
    for i in range(0, Lapp_old.shape[0]):
        if Lapp_old[i][8]< mag_lim:
            Lapp.append(Lapp_old[i])
    Lapp = np.asarray(Lapp)
    Lapp_err = np.full((len(Lapp),n_bands),0.01) # Create array with magnitude errors

    flux = mag_to_jansky(Lapp)
    flux_err = flux - mag_to_jansky(Lapp + Lapp_err)

    # Adding error floors due to systematic errors in filters
    irac_bands = [9,10]
    flux_err = flux_err + 0.05*flux # 0.05 for all
    for i in range(0, len(irac_bands)):
        flux_err[:,irac_bands[i]] = flux_err[:,irac_bands[i]] + 0.2*flux[:,irac_bands[i]] # 0.2 for the IRAC bands

    # Create array with redshifts
    z = np.full(len(Lapp[0]), redshift)

    # Get magnitude of selection filter
    Kmag = Lapp[:,ind_select]

    return caesar_id, flux, flux_err, z, Kmag

def fill_flux(flux, z, minz, maxz, dz, ll_obs, ind):
    '''
    Place f_nu_obs into super-sampled array and then convert into f_lambda_rest.
    '''
    nredshift = int((maxz-minz)/dz) + 1
    zbin = np.linspace(minz, maxz, nredshift)
    nz = len(zbin)
    print(nz)
    n_band = len(ll_obs)
    ff = c_in_AA * flux / (ll_obs**2) # f_nu_obs to f_lambda_obs
    ff = ff * (1+z) # f_lambda_obs to f_lambda_rest

    # Find into which redshift bin the galaxy lands
    tmp = abs(z - zbin)
    ind_zz = tmp.argmin()

    # Find into which band bin to put flux into
    fluxarr = np.zeros((n_band,nz))
    for i in range(0, n_band):
        fluxarr[i][ind_zz] = ff[i]

    ind_select = [[],[]]
    for i in range(0, len(ind)):
        row = ind[i]%nz
        column = int(ind[i]/nz)
        ind_select[0].append(row)
        ind_select[1].append(column)

    fluxarr = fluxarr[ind_select]

    return fluxarr


def superflux(minz, manz, dz, ind, wave, flux, flux_err, z, ll_eff):
    '''
    Calculate rest-frame f_lambda and put into correct PCA supergrid
    '''
    ngal = len(flux[0])
    nband = len(ind)
    flux_super = np.zeros((ngal, nband))
    flux_super_err = np.zeros((ngal, nband))

    for i in range(0, ngal):
        flux_super[i] = fill_flux(flux[i],z[i],minz,maxz,dz,ll_eff,ind)
        flux_super_err[i] = fill_flux(flux_err[i],z[i],minz,maxz,dz,ll_eff,ind)

    return flux_super, flux_super_err


wave,spec,mean,var,ind,minz,maxz,dz,filternames,ll_eff = read_eigensystem('../VWSC_simba/EBASIS/VWSC_eigenbasis_0p5z3_wavemin2500.fits', '../VWSC_simba/FILTERS/vwsc_uds.lis')
ind_filt = [0,1,2,3,4,5,6,7,8,11,12]
n_bands = len(ll_eff)
caesar_id, flux, flux_err, z, Kmag = data_from_simba('/home/rad/data/m100n1024/s50/Groups/phot_m100n1024_026.hdf5', n_bands, 29.5, ind_filt, 8)

ll_obs = ll_eff[ind_filt]

flux_super, flux_super_err = superflux(minz, maxz, dz, ind, wave, flux, flux_err, z, ll_eff)

print(flux_err)
print(flux_super_err)
