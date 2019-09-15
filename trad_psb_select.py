#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13 August 2019

@author: Curro Rodriguez Montero, School of Physics and Astronomy,
            University of Edinburgh, JCMB, King's Buildings

This codes obtains the equivalent width for a set of particular lines used in traditional PSBs
slection techniques; all from the galactic spectra output of the pyloser code.

For questions about the code:
s1650043@ed.ac.uk
"""
# Import necessary libraries
import numpy as np 
import h5py
import sys
import caesar 
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

MODEL = sys.argv[1]
WIND = sys.argv[2]
SNAP = int(sys.argv[3])
#GALAXY = [int(sys.argv[4])]

def EW_hdelta(flux,waves):
    hd_window = np.array([4082,4122]) # in Angstroms, as given by Goto et al. (2003)
    blue_window = np.array([4030,4082])
    red_window = np.array([4122,4170])
    ind_start = np.argmin(abs(waves-hd_window[0]))
    ind_end = np.argmin(abs(waves-hd_window[1]))
    blue = np.median(flux[np.argmin(abs(waves-blue_window[0])):np.argmin(abs(waves-blue_window[1]))])
    blue_w = (waves[np.argmin(abs(waves-blue_window[0]))]+waves[np.argmin(abs(waves-blue_window[1]))])/2
    red = np.median(flux[np.argmin(abs(waves-red_window[0])):np.argmin(abs(waves-red_window[1]))])
    red_w = (waves[np.argmin(abs(waves-red_window[0]))]+waves[np.argmin(abs(waves-red_window[1]))])/2
    f = flux[ind_start:ind_end+1]
    w = waves[ind_start:ind_end+1]
    w_p = np.array([blue_w,red_w])
    f_p = np.array([blue,red])
    f_0 = np.interp(w,w_p,f_p)
    d = abs(w[1] - w[0])
    W = np.sum((1-f/f_0)*d)
    w_p = np.array([w[0],w[-1]])
    return W, w_p

def EW_halpha(flux,waves):
    hd_window = np.array([6555,6575]) # in Angstroms, as given by Goto et al. (2003)
    blue_window = np.array([6490,6537])
    red_window = np.array([6594,6640])
    ind_start = np.argmin(abs(waves-hd_window[0]))
    ind_end = np.argmin(abs(waves-hd_window[1]))
    blue = np.median(flux[np.argmin(abs(waves-blue_window[0])):np.argmin(abs(waves-blue_window[1]))])
    blue_w = (waves[np.argmin(abs(waves-blue_window[0]))]+waves[np.argmin(abs(waves-blue_window[1]))])/2
    red = np.median(flux[np.argmin(abs(waves-red_window[0])):np.argmin(abs(waves-red_window[1]))])
    red_w = (waves[np.argmin(abs(waves-red_window[0]))]+waves[np.argmin(abs(waves-red_window[1]))])/2
    f = flux[ind_start:ind_end+1]
    w = waves[ind_start:ind_end+1]
    w_p = np.array([blue_w,red_w])
    f_p = np.array([blue,red])
    f_0 = np.interp(w,w_p,f_p)
    d = abs(w[1] - w[0])
    W = np.sum((1-f/f_0)*d)
    w_p = np.array([w[0],w[-1]])
    return W, w_p

def plot_spectra(flux,waves,gal,snap,model):
    # Just do a simple spectra plot for the input of flux and wavelenghts given.
    fig = plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.plot(waves,flux,'k-')
    Wd, w_pd = EW_hdelta(flux,waves)
    Wa, w_pa = EW_halpha(flux,waves)
    ax.text(0.05,0.8,r'EW(H$_{\delta}$) = '+'{:.2}'.format(Wd)+'$\AA$',transform=ax.transAxes)
    ax.axvspan(w_pd[0],w_pd[1],alpha=0.6)
    ax.text(0.4,0.6,r'EW(H$_{\alpha}$) = '+'{:.2}'.format(Wa)+'$\AA$',transform=ax.transAxes)
    ax.axvspan(w_pa[0],w_pa[1],alpha=0.6)
    ax.set_xlabel(r'$\lambda [\AA$]', fontsize=16)
    ax.set_ylabel('Flux',fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/'+str(model)+'/spectra_'+str(gal)+'_'+str(snap)+'.png',format='png', dpi=250, bbox_inches='tight')
def read_pyloser(model,wind,snap):
    
    loser_file = '/home/rad/data/%s/%s/Groups/loser_%s_%03d.hdf5' % (model,wind,model,snap)
    f = h5py.File(loser_file,'r')
    wave = np.asarray(f['myspec_wavelengths'][:])
    ids = f['iobjs'][:]
    #fluxes = np.zeros((len(f['iobjs'][:]),len(wavelengths)))
    fluxes = []
    wavelengths = []
    c_file = '/home/rad/data/%s/%s/Groups/%s_%03d.hdf5' % (model,wind,model,snap)
    sim = caesar.load(c_file, LoadHalo=False)
    redshift = sim.simulation.redshift  # this is the redshift of the simulation output
    h = sim.simulation.hubble_constant  # this is the hubble parameter = H0/100
    cosmo = FlatLambdaCDM(H0=100*sim.simulation.hubble_constant, Om0=sim.simulation.omega_matter, Ob0=sim.simulation.omega_baryon,Tcmb0=2.73)  # set our cosmological parameters
    H = cosmo.H(redshift).to('km/(kpc s)').value
    thubble = cosmo.age(redshift).value
    ssfr_limit = np.log10(0.2/(thubble))-9
    for i in range(0,len(f['iobjs'][:])):
        if (sim.galaxies[int(ids[i])].sfr/sim.galaxies[int(ids[i])].masses['stellar']) >= 10**ssfr_limit:
            #fluxes[i,:] = f['myspec'][i,:]
            #print(sim.galaxies[int(ids[i])].vel[0],sim.galaxies[int(ids[i])].pos[0])
            vx = float(sim.galaxies[int(ids[i])].vel[0].value) + float(H)*float(sim.galaxies[int(ids[i])].pos[0].to('kpc')/h)
            wavelengths.append(wave/(1+vx/(299792.458)))
            fluxes.append(f['myspec'][i,:])
    fluxes = np.asarray(fluxes)
    wavelengths = np.asarray(wavelengths)
    return wavelengths,fluxes

def halpha_hdelta_plot(wave,flux,model,snap):
    ngals = flux.shape[0]
    halpha = []
    hdelta = []
    for i in range(0,ngals):
        W_a, w_p = EW_halpha(flux[i,:],wave[i,:])
        W_d, w_p = EW_hdelta(flux[i,:],wave[i,:])
        #if W_a>=-2.5 and W_d>=0:
        if W_d>=0:
            halpha.append(W_a)
            hdelta.append(W_d)
    halpha = np.asarray(halpha)
    hdelta = np.asarray(hdelta)
    fig = plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.hexbin(hdelta,halpha,bins='log', cmap='Greys')
    ax.set_xlabel(r'EW(H$_{\delta}) [\AA$]', fontsize=16)
    ax.set_ylabel(r'EW(H$_{\alpha}) [\AA$]', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/'+str(model)+'/hahd_'+str(snap)+'.png',format='png', dpi=250, bbox_inches='tight')



wave,flux = read_pyloser(MODEL,WIND,SNAP)
halpha_hdelta_plot(wave,flux,MODEL,SNAP)
plot_spectra(flux[0,:],wave[0,:],0,SNAP,MODEL)