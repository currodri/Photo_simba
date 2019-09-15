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
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt



#Imports needed for the paralelization
import sys
import multiprocessing

#Set the number of processors to use. It can be defined passing an argument through 
#the command line or taking the maximum number of cpus available in the system.
# if len(sys.argv) > 1:
#     num_proc = int(sys.argv[1])
# else:
#     num_proc = multiprocessing.cpu_count()
num_proc = multiprocessing.cpu_count()

## CONSTANTS
c_in_AA = 2.99792*(10**18)
sol_lum_in_erg = 3.839*(10**33)
pc_in_m = 3.08567758*(10**16)
Mpc_in_cm = 3.08567758*(10**23)
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
        print('Maximum redshift of the survey used: '+str(maxz))

    # Read in filter effective wavelengths into array
    f = open(filterfile).readlines()
    ll_eff = np.zeros(len(f))
    for i in range(0, len(f)):
        ll_eff[i] = float(f[i].split()[1])
    return wave,spec,mean,var,ind,minz,maxz,dz,filternames,ll_eff

def mag_to_jansky(mag_AB):
    '''
    From AB magnitude array, returns the equivalent array of the flux in units
    of Jansky (1Jy = 10-23 erg s-1 Hz-1 cm-2).
    '''
    f_nu_ergs = 10**(-0.4*(mag_AB + 48.6))
    f_nu = f_nu_ergs*(10**23)

    return f_nu

def data_from_closer(ph_file, n_bands, mag_lim, ind_filt, ind_select):
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
    print(len(caesar_id))
    header = f['HEADER_INFO']
    colorinfo = f['COLOR_INFO'][:]
    redshift = float(header[0].split()[2]) # Get redshift of snapshot
    Lapp_old = np.zeros((len(caesar_id),n_bands))
    ind = [34,35,36,37,38,39,40,41,42,17,18]
    # Apparent magnitudes of galaxies in each desired band
    for (i,i_filt) in zip(ind, ind_filt):
        Lapp_old[:,i_filt] = f['appmag_%d'%i] # Save mags for the selected filters
        print ('Reading now filter for '+str(colorinfo[i]))
    # Apply magnitude limit given by mag_lim
    Lapp = []
    for i in range(0, Lapp_old.shape[0]):
        if Lapp_old[i][8]< mag_lim:
            Lapp.append(Lapp_old[i])
    Lapp = np.asarray(Lapp)
    Lapp_err = np.full((len(Lapp),n_bands),0.01) # Create array with magnitude errors
    flux = mag_to_jansky(Lapp)
    flux_err = flux - mag_to_jansky(Lapp + Lapp_err)
    ind_ignore = np.where(Lapp==0)
    for i in range(0,len(ind_ignore[0])):
        a = ind_ignore[0][i]
        b = ind_ignore[1][i]
        flux[a,b] = 0.0
        flux_err[a,b] = 0.0

    # Adding error floors due to systematic errors in filters
    irac_bands = [11,12]
    for i in range(0,flux_err.shape[0]):
        for j in range(0,flux_err.shape[1]):
            if flux_err[i,j] < 0.05*flux[i,j]:
                flux_err[i,j] = 0.05*flux[i,j]
    for i in range(0, len(irac_bands)):
        for j in range(0,flux_err.shape[0]):
            if flux_err[j,irac_bands[i]] < 0.2*flux[j,irac_bands[i]]:
                flux_err[j,irac_bands[i]] = 0.2*flux[j,irac_bands[i]] # 0.2 for the IRAC bands

    # Create array with redshifts
    z = np.full(Lapp.shape[0], redshift)

    # Get magnitude of selection filter
    Kmag = Lapp[:,ind_select]

    return caesar_id, flux, flux_err, z, Kmags

def data_from_pyloser(loser_file, n_bands, mag_lim, ind_filt, ind_select, redshift):
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
    f = h5py.File(loser_file,'r') # Read in .hdf5 file with photometry catalogue
    caesar_id = f['iobjs'][:]
    Lapp_old = np.zeros((len(caesar_id),n_bands))
    
    # The array ind holds the selected filters indexes from the loser catalogue which match
    # the filters used for the eigensystem. MAKE SURE THEY ARE EXACTLY THE SAME

    #ind = [34,35,36,37,38,39,40,41,42,17,18]
    #ind = [1,3,4,5,6,7,8,9,10,11,12]
    ind = [22,23,25,26,27,28,10,11,12,13,14]
    # Apparent magnitudes of galaxies in each desired band
    for (i,i_filt) in zip(ind, ind_filt):
        Lapp_old[:,i_filt] = f['mymags'][:,i] # Save mags for the selected filters
    # Apply magnitude limit given by mag_lim
    Lapp = []
    for i in range(0, Lapp_old.shape[0]):
        if Lapp_old[i][8]< mag_lim:
            Lapp.append(Lapp_old[i])
    Lapp = np.asarray(Lapp)
    Lapp_err = np.full((len(Lapp),n_bands),0.01) # Create array with magnitude errors
    flux = mag_to_jansky(Lapp)
    flux_err = flux - mag_to_jansky(Lapp + Lapp_err)
    ind_ignore = np.where(Lapp==0)
    for i in range(0,len(ind_ignore[0])):
        a = ind_ignore[0][i]
        b = ind_ignore[1][i]
        flux[a,b] = 0.0
        flux_err[a,b] = 0.0

    # Adding error floors due to systematic errors in filters
    irac_bands = [11,12]
    for i in range(0,flux_err.shape[0]):
        for j in range(0,flux_err.shape[1]):
            if flux_err[i,j] < 0.05*flux[i,j]:
                flux_err[i,j] = 0.05*flux[i,j]
    for i in range(0, len(irac_bands)):
        for j in range(0,flux_err.shape[0]):
            if flux_err[j,irac_bands[i]] < 0.2*flux[j,irac_bands[i]]:
                flux_err[j,irac_bands[i]] = 0.2*flux[j,irac_bands[i]] # 0.2 for the IRAC bands

    # Create array with redshifts
    z = np.full(Lapp.shape[0], redshift)

    # Get magnitude of selection filter
    Kmag = Lapp[:,ind_select]

    return caesar_id, flux, flux_err, z, Kmag

def fill_flux(args):
    '''
    Place f_nu_obs into super-sampled array and then convert into f_lambda_rest.
    '''
    #Unpack the tuple of arguments
    flux, z, minz, maxz, dz, ll_obs, ind = args

    
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
    ind_select = tuple(ind_select)
    fluxarr = fluxarr[ind_select]

    return fluxarr

def superflux(minz, maxz, dz, ind, wave, flux, flux_err, z, ll_eff,p_workers):
    '''
    Calculate rest-frame f_lambda and put into correct PCA supergrid
    '''
    ngal = flux.shape[0]
    n_band = len(ind)
    flux_super = np.zeros((ngal, n_band))
    flux_super_err = np.zeros((ngal, n_band))
    flux_super = np.array([])

    #List of tuples being each tuple the argument that is going to be passed
    #to the function fill_flux
    args = [(flux[i],z[i],minz,maxz,dz,ll_eff,ind) for i in range(ngal)]
    #Compute the fill_flux with the different arguments and store it in a np array.
    flux_super = np.array(p_workers.map(fill_flux, args))

    #Do the same but changing the arguments
    args = [(flux_err[i],z[i],minz,maxz,dz,ll_eff,ind) for i in range(ngal)]
    flux_super_err = np.array(p_workers.map(fill_flux, args))


    return flux_super, flux_super_err

def norm_gappy(args):
    # Unpack the arguments of the function
    nrecon, nbin, error, data_j, mean, espec, verbose, cov = args
    if verbose:
            print('[pca_normgappy] STATUS: processing spectrum ')

    # Calculate weighting array from 1-sig error array
    # ! if all bins have error=0 continue to next spectrum
    weight = np.zeros(nbin)
    ind = error.nonzero()[0]
    if np.size(ind) != 0:
        try:
            weight[ind] = 1. / (error[ind]**2)
        except:
            if verbose:
                print(
                    '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                )

    ind = np.where(np.isfinite(weight) is False)[0]
    if np.size(ind) != 0:
        if verbose:
            print(
                '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
            )

    # Solve partial chi^2/partial N = 0
    Fpr = np.sum(weight * data_j * mean)  # eq 4 [2]
    Mpr = np.sum(weight * mean * mean)  # eq 5 [2]
    E = np.sum((weight * mean) * espec, axis=1)  # eq 6 [2]

    # Calculate the weighted eigenvectors, multiplied by the eigenvectors (eq. 4-5 [1])
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

    pcs = np.squeeze(np.sum(Fnew * Minv, 1))
    norm = Fpr / (Mpr + np.sum(pcs * E))

    # Calculate covariance matrix (eq. 6 [1])
    if cov is True:
        M_gappy = np.dot((espec * (weight * norm**2)), espec.T)
        ccov = np.linalg.inv(M_gappy)
        return pcs, norm, ccov
    else:
        return pcs, norm

def parall_normgappy(data, error, espec, mean,p_workers, cov=False,verbose=False):
    '''
    This is the main code for Super Color Analysis: the spectrum vectors of the
    galaxies are projected onto the eigenbasis and normalized.

    Based on the original code in IDL by Vivienne Wild.
    '''

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

    #List of tuples being each tuple the argument that is going to be passed
    #to the function norm_gappy
    args = [(nrecon, nbin, error[j,:], data[j,:], mean, espec, cov, verbose) for j in range(ngal)]
    results = np.array(p_workers.map(norm_gappy,args))

    for i in range(0, len(results)):
        if ccov is True:
            pcs[i,:] = results[i][0]
            norm[i] = results[i][1]
            ccov[i,:,:] = results[i][2]
        else:
            pcs[i,:] = results[i][0]
            norm[i] = results[i][1]

    if ngal == 1:
        pcs = pcs[0]
        data = data[0]
        norm = norm[0]
        if cov:
            ccov = ccov[0]

    # Return

    if cov is True:
        return pcs, norm, ccov
    else:
        return pcs, norm
        
def SC1_vs_SC2_scatter(pc_data,snap,lines):
    '''
    Example code that uses the SC1 and SC2 amplitudes to make a scatter plot.
    '''

    x = pc_data[:,0] #SC1
    y = pc_data[:,1] #SC2
    print(len(x), len(y))
    fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel('SC 2', fontsize=16)
    ax.set_xlabel('SC 1', fontsize=16)
    ax.scatter(x,y, s=10)

    # Set classification lines in SC diagram
    x_red = np.linspace(1.0,25.0,25) - 35.
    y_red = lines['red_sl']*x_red+lines['red_int']     
    ax.plot(x_red, y_red, 'k-')
        
    x_psb = np.linspace(1.0,40.0,40) - 10.
    y_psb = lines['psb_sl']*x_psb+lines['psb_int']
    ax.plot(x_psb,y_psb)

    x_dusty = np.linspace(1.0,25.0,25) - 35.
    y_dusty = lines['dusty_sl']*x_dusty+lines['dusty_int']
    ax.plot(x_dusty,y_dusty)

    x_psb2 = np.linspace(1.0,25.0,25) - 25.
    y_psb2 = lines['psb2_sl']*x_psb2+lines['psb2_int']
    ax.plot(x_psb2, y_psb2)


    ax.set_xlim([-50,150])
    ax.set_ylim([-20, 30])
    fig.tight_layout()
    fig.savefig('../color_plots/sc1_vs_sc2_'+str(snap)+'.png',
                    format='png', dpi=250, bbox_inches='tight')

def point_seg_distance(point, seg_start, seg_end, interval=True):
    '''
    Simple code that obtains the geometric distance between a point and a segment.
    If the point is outside of the interval spanned by the segment, the distance
    to the closest point is returned.
    '''

    lv = seg_start - seg_end
    l = np.sqrt(np.dot(lv,lv))

    lv = lv/l 
    t = - (-np.dot(lv,point) + np.dot(seg_start,lv))/(np.dot(lv,lv))
    pl = t * lv + seg_start
    out = t < 0 or t > l

    if interval and out:
        d1 = np.sqrt(np.sum((point-seg_start)**2))
        d2 = np.sqrt(np.sum((point-seg_end)**2))
        ans = np.amin(np.array([d1,d2]))
        if point[0] < pl[0]:
            ans = - ans 
    else:
        ans = np.sqrt(np.sum((point-pl)**2))
        if point[0] < pl[0]:
            ans = - ans
    return ans,pl[0]


def SC_classification(pcs, lines, plot=True):

    '''

    NOT USE!! Delimitation lines are still not properly adjusted. Will be corrected in future versions.

    This code provides a classification of the continuum distribution of SC of 
    a set of galaxies. It depends on the eigenbasis used, so make sure you're using
    the right one for the input set in read_eigensystem.

    The original code was written in IDL by Vivienne Wild.

    pcs == array containing the SC amplitudes for the galaxies
    lines == dictionary containing the set of intercepts and slope of the lines that
                subdivide the disttribution.
            An example of the format would be like this:
            {'red_sl':0.783, 'red_int':14.83, 'psb_sl':0.4,'psb_int':10.86, 'psb2_sl':-0.34,'psb2_int':3.19,
                'dusty_sl':-0.2, 'dusty_int':-13.75,'lomet_pc3lo':3.5, 'lomet_sl':0, 'lomet_pc1hi':10, 'green_cut':0, 'dust_cut':-20 }
    plot == plot SC1 vs SC2 diagram with the classification overimposed. Default is True.
    '''
    ngal = pcs.shape[0]
    pcs = np.asarray(pcs)

    # Set classification lines in SC diagram
    x_red = np.linspace(1.0,25.0,25) - 35.
    y_red = lines['red_sl']*x_red+lines['red_int']     
        
    x_psb = np.linspace(1.0,40.0,40) - 10.
    y_psb = lines['psb_sl']*x_psb+lines['psb_int']

    x_dusty = np.linspace(1.0,25.0,25) - 35.
    y_dusty = lines['dusty_sl']*x_dusty+lines['dusty_int']

    x_psb2 = np.linspace(1.0,25.0,25) - 25.
    y_psb2 = lines['psb2_sl']*x_psb2+lines['psb2_int']

    # Distance from and along PSB line
    dist_psb = np.zeros(ngal)
    xy_psb = np.zeros(ngal)
    seg_start = np.array([min(x_psb),min(y_psb)])
    seg_end = np.array([max(x_psb),max(y_psb)])
    for i in range(0, ngal):
        dist_psb[i],xy_psb[i] = point_seg_distance(pcs[i][0:2],seg_start,seg_end)

    # Distance from and along red line
    dist_red = np.zeros(ngal)
    xy_red = np.zeros(ngal)
    seg_start = np.array([min(x_red),min(y_red)])
    seg_end = np.array([max(x_red),max(y_red)])
    for i in range(0, ngal):
        dist_red[i],xy_red[i] = point_seg_distance(pcs[i][0:2],seg_start,seg_end)

    # Distance from and along dusty line
    dist_dusty = np.zeros(ngal)
    xy_dusty = np.zeros(ngal)
    seg_start = np.array([x_dusty[0],y_dusty[0]])
    seg_end = np.array([x_dusty[24],y_dusty[24]])
    for i in range(0, ngal):
        dist_dusty[i],xy_dusty[i] = point_seg_distance(pcs[i][0:2],seg_start,seg_end)

    # Distance from and along PSB 2 line
    dist_psb2 = np.zeros(ngal)
    xy_psb2 = np.zeros(ngal)
    seg_start = np.array([x_psb2[0],y_psb2[0]])
    seg_end = np.array([x_psb2[24],y_psb2[24]])
    for i in range(0, ngal):
        dist_psb2[i],xy_psb2[i] = point_seg_distance(pcs[i][0:2],seg_start,seg_end)

    # Find Red Sequence
    ind_red  = np.where(dist_red < 0. and dist_dusty > 0. and dist_psb2 < 0.)
    ind_dusty = np.where(pcs[:,0] < lines['dust_cut'] and dist_dusty < 0.)
    ind_dusty = np.isin(ind_dusty, ind_red)
    ind_psb = np.where(dist_psb < 0. and dist_psb2 > 0.)

    # Find Blue Cloud
    ind_sf3 = np.where(pcs[:,0] < lines['green_cut'])
    ind_sf3 = np.isin(ind_sf3, ind_dusty)
    ind_sf3 = np.isin(ind_sf3, ind_red)
    ind_sf3 = np.isin(ind_sf3, ind_psb)

    ind_sf1 = np.where(pcs[:,0] > lines['sb_cut'])
    ind_sf1 = np.isin(ind_sf1, ind_psb)
    ind_sf1 = np.isin(ind_sf1, ind_dusty)

    ind_sf2 = np.where(pcs[:,0] >= lines['green_cut'] and pcs[:,0] <= lines['sb_cut'])
    ind_sf2 = np.isin(ind_sf2, ind_dusty)
    ind_sf2 = np.isin(ind_sf2, ind_psb)
    ind_sf2 = np.isin(ind_sf2, ind_red)

    lometint = lines['lomet_pc3lo']
    lometsl = lines['lomet_sl']

    # Find Low Metallicity
    ind_lomet = np.where(pcs[:,2] > lometsl*pcs[:,0]+lometint)
    ind_lomet = np.isin(ind_lomet, ind_sf3)
    ind_lomet = np.isin(ind_lomet, ind_sf2)
    ind_lomet = np.isin(ind_lomet, ind_sf1)
    ind_lomet = np.isin(ind_lomet, ind_dusty)

    ind_red = np.isin(ind_red, ind_lomet)
    ind_psb = np.isin(ind_psb, ind_lomet)
    ind_dusty = np.isin(ind_dusty, ind_lomet)

    # Get rind of the rubbish
    #ind_junk = np.where(pcs[:,0]==0.)


def main(lfile, caesar_redshift):
    '''
    Main routine for the parallelization of the SCA code for a given loser catologue file.

    Make sure the paths to the eigenbasys and filters are correct.
    '''
    wave,spec,mean,var,ind,minz,maxz,dz,filternames,ll_eff = read_eigensystem('/home/curro/quenchingSIMBA/code/photo/VWSC_simba/EBASIS/VWSC_eigenbasis_0p5z3_wavemin2500.fits', '/home/curro/quenchingSIMBA/code/photo/VWSC_simba/FILTERS/vwsc_uds.lis')
    ind_filt = [0,1,2,3,4,5,6,7,8,11,12]
    n_bands = len(ll_eff)
    caesar_id, flux, flux_err, z, Kmag = data_from_pyloser(lfile, n_bands, 300, ind_filt, 8, caesar_redshift)
    
    #Let's create a pool of workers, that is a set of processes that we can use to handle computation.
    p_workers = multiprocessing.Pool(num_proc)
    
    ll_obs = ll_eff[ind_filt]

    flux_super, flux_super_err = superflux(minz, maxz, dz, ind, wave, flux, flux_err, z, ll_eff,p_workers)
    # Usually, limiting the analysis to thw first three eigenspectra is sufficient to hold the majority
    # of the population variability.
    pcs, norm = parall_normgappy(flux_super/(10**5),flux_super_err/(10**5),spec[0:3,:],mean,p_workers)

    pcs = np.asarray(pcs)

    return pcs

