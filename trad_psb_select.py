#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13 August 2019

@author: Curro Rodriguez Montero, School of Physics and Astronomy,
            University of Edinburgh, JCMB, King's Buildings

This codes obtains the equivalent width for a set of particular lines from the galactic spectra output
of the pyloser code.

For questions about the code:
s1650043@ed.ac.uk
"""
# Import necessary libraries
import numpy as np 
import h5py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

MODEL = sys.argv[1]
WIND = sys.argv[2]
SNAP = int(sys.argv[3])
GALAXY = int(sys.argv[4])

def plot_spectra(flux,waves,gal,snap):
    # Just do a simple spectra plot for the input of flux and wavelenghts given.

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.plot(waves,flux,'k-')
    ax.set_xlabel(r'$\lambda [10^{-10}$ m]', fontsize=16)
    ax.set_ylabel('Flux',fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/spectra_'+str(gal)+'_'+str(snap)+'.png',format='png', dpi=250, bbox_inches='tight')
def read_pyloser(model,wind,snap,gals):

    loser_file = '/home/rad/data/%s/%s/Groups/loser_%s_%03d.hdf5' % (model,wind,model,snap)
    f = h5py.File(loser_file,'r')
    wavelengths = np.asarray(f['myspec_wavelengths'][:])
    fluxes = np.zeros((len(gals),len(wavelengths)))
    for i in range(0,len(gals)):
        fluxes[i,:] = f['myspec'][int(gals[i]),:]
    return wavelengths,fluxes

wave,flux = read_pyloser(MODEL,WIND,SNAP,GALAXY)
plot_spectra(flux[0,:],wave,GALAXY,SNAP)