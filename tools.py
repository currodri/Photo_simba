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
        column = ind[i]%nz
        row = int(ind[i]/nz)
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

def normgappy(data, error, espec, mean, cov=False, reconstruct=False, verbose=False):
    """
    Performs robust PCA projection, including normalization estimation.
    Parameters
    ----------
    data : ndarray
        1D spectrum or 2D specta with 'float' type.
    error : ndarray
        1D or 2D corresponding 1-sigma error array. Zeros indicate masked data.
    espec : ndarray
        2D array of eigenspectra, possibly truncated in dimension.
    mean : ndarray
        1D mean spectrum of the eigenspectra.
    cov: bool, optional
        Return covariance matrix.
        Default is ''False''.
    reconstruct : bool, optional
        Fill in missing values with PCA estimation.
        Default is ''False''.
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''
    Returns
    -------
    pcs : ndrray
        1D or 2D array of Principal Components with 'float' type.
    norm: float or ndarray
        Normalization estimates.
    data : ndrray
        If reconstruct enabled, 1D or 2D reconstructed spectra.
    ccov : ndarray
        If cov enabled, 2D or 3D covariance matrices.
    """

    # Sanity checks
    if (np.size(data) == 0) | (np.size(error) == 0) | (np.size(espec) == 0) | (
            np.size(mean) == 0):
        print('[pca_normgappy] ERROR: incorrect input lengths')
        return None

    tmp = np.shape(espec)  # number of eigenvectors
    if np.size(tmp) == 2:
        nrecon = tmp[0]
    else:
        nrecon = 1
    nbin = np.shape(espec)[-1]  # number of data points
    tmp = np.shape(data)  # number of observations to project
    if np.size(tmp) == 2:
        ngal = tmp[0]
    else:
        ngal = 1

    # Dimension mismatch check
    if np.shape(data)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "data" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(error)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "error" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(mean)[0] != nbin:
        print(
            '[pca_normgappy] ERROR: "mean" must have the same dimension as eigenvectors'
        )
        return None

    # Project each galaxy in turn
    pcs = np.zeros((ngal, nrecon), float)
    norm = np.zeros(ngal, float)
    if cov is not None:
        ccov = np.zeros((ngal, nrecon, nrecon))

    if ngal == 1:
        data = data[np.newaxis, :]
        error = error[np.newaxis, :]

    for j in np.arange(0, ngal):

        if verbose:
            print('[pca_normgappy] STATUS: processing spectrum ')

        # Calculate weighting array from 1-sig error array
        # ! if all bins have error=0 continue to next spectrum
        weight = np.zeros(nbin)
        ind = error[j, :].nonzero()[0]
        if np.size(ind) != 0:
            try:
                weight[ind] = 1. / (error[j, :][ind]**2)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                    )
                continue

        ind = np.where(np.isfinite(weight) is False)[0]
        if np.size(ind) != 0:
            if verbose:
                print(
                    '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                )
            continue

        data_j = data[j, :]

        # Solve partial chi^2/partial N = 0
        Fpr = np.sum(weight * data_j * mean)  # eq 4 [2]
        Mpr = np.sum(weight * mean * mean)  # eq 5 [2]
        E = np.sum((weight * mean) * espec, axis=1)  # eq 6 [2]

        # Calculate the weighted eigenvectors, multiplied by the eigenvectors (eq. 4-5 [1])

        if nrecon > 1:
            # CONSERVED MEMORY NOT IMPLEMETED
            espec_big = np.repeat(espec[:, np.newaxis, :], nrecon, axis=1)
            M = np.sum(weight * np.transpose(espec_big, (1, 0, 2)) * espec_big, 2)

            # Calculate the weighted data array, multiplied by the eigenvectors (eq. 4-5 [1])
            F = np.dot((data_j * weight), espec.T)

            # Calculate new M matrix, this time accounting for the unknown normalization (eq. 11 [2])
            E_big = np.repeat(E[np.newaxis, :], nrecon, axis=0)
            F_big = np.repeat(F[:, np.newaxis], nrecon, axis=1)
            Mnew = Fpr * M - E_big * F_big

            # Calculate the new F matrix, accounting for unknown normalization
            Fnew = Mpr * F - Fpr * E

            # Solve for Principle Component Amplitudes (eq. 5 [1])
            try:
                Minv = np.linalg.inv(Mnew)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] STATUS: problem with matrix inversion (setting pcs=0)'
                    )

                continue

            pcs[j, :] = np.squeeze(np.sum(Fnew * Minv, 1))
            norm[j] = Fpr / (Mpr + np.sum(pcs[j, :] * E))

            # Calculate covariance matrix (eq. 6 [1])
            if cov is True:
                M_gappy = np.dot((espec * (weight * norm[j]**2)), espec.T)
                ccov[j, :, :] = np.linalg.inv(M_gappy)

        else:  # if only one eigenvector
            M = np.sum(weight * espec * espec)
            F = np.sum(weight * data_j * espec)
            Mnew = M * Fpr - E * F
            Fnew = Mpr * F - E * Fpr
            pcs[j, 0] = Fnew / Mnew
            norm[j] = Fpr / (Mpr + pcs[j, 0] * E)
            if cov is True:
                ccov[j, 0, 0] = np.sum((1. / weight) * espec * espec)

        # If reconstruction of data array required,
        #   fill in regions with weight = 0 with PCA reconstruction
        if reconstruct is True:
            bad_pix = np.where(weight == 0.)
            count = np.size(bad_pix)
            if count == 0:
                continue

            rreconstruct = np.sum((pcs[j, :] * espec[:, bad_pix].T).T, 0)
            rreconstruct += mean[bad_pix]
            data[j, bad_pix] = reconstruct

    if ngal == 1:
        pcs = pcs[0]
        data = data[0]
        norm = norm[0]
        if cov:
            ccov = ccov[0]

    # Report to user
    if verbose:
        print("[pca_normgappy] STATUS: Results...")
        for i, pc in enumerate(pcs):
            print(f"               PCA{i+1}: {pc:2.5f}")
        print(f"               Norm: {norm:2.5f}")

    # Return
    if reconstruct is True:
        if cov is True:
            return pcs, norm, data, ccov
        else:
            return pcs, norm, data

    elif cov is True:
        return pcs, norm, ccov
    else:
        return pcs, norm

def SC1_vs_SC2_scatter(pc_data):

    x = pc_data[:,0] #SC1
    y = pc_data[:,1] #SC2
    fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel('SC 2', fontsize=16)
    ax.set_xlabel('SC 1', fontsize=16)
    ax.scatter(x,y)
    fig.tight_layout()
    fig.savefig('../color_plots/sc1_vs_sc2.png',
                    format='png', dpi=250, bbox_inches='tight')
wave,spec,mean,var,ind,minz,maxz,dz,filternames,ll_eff = read_eigensystem('../VWSC_simba/EBASIS/VWSC_eigenbasis_0p5z3_wavemin2500.fits', '../VWSC_simba/FILTERS/vwsc_uds.lis')
ind_filt = [0,1,2,3,4,5,6,7,8,11,12]
n_bands = len(ll_eff)
caesar_id, flux, flux_err, z, Kmag = data_from_simba('/home/rad/data/m100n1024/s50/Groups/phot_m100n1024_026.hdf5', n_bands, 29.5, ind_filt, 8)

ll_obs = ll_eff[ind_filt]

flux_super, flux_super_err = superflux(minz, maxz, dz, ind, wave, flux, flux_err, z, ll_eff)

pcs = normgappy(flux_super,flux_super_err,spec,mean)

SC1_vs_SC2_scatter(pcs)
