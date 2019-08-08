#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 August 2019

@author: Curro Rodriguez Montero, School of Physics and Astronomy,
            University of Edinburgh, JCMB, King's Buildings

This code is a case example in which the position of a given galaxy is tracked in colour space. It uses the results stored
in the pickle file coming from the mergerFinder and quenchingFinder algorithms, as well the photometry data from the closer
routine.

For questions about the code:
s1650043@ed.ac.uk
"""

# Import required libraries
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import cPickle as pickle
import sys
sys.path.insert(0, '../../SH_Project')
from galaxy_class import Magnitude, GalaxyData

# Get details from terminal

MODEL = sys.argv[1]     # e.g. m100n1024
REDSHIFT = float(sys.argv[2])  # e.g. 0.5
GALAXY = [int(sys.argv[3])]    # e.g. 987
if len(sys.argv) > 4:
    GALAXY = sys.argv[4:] # this is for the case in which we want the tracks for multiple galaxies

# Read data from pickle file
data_file = '/home/curro/quenchingSIMBA/code/SH_Project/mandq_results_%s.pkl' % (MODEL)
obj = open(data_file, 'rb')
d = pickle.load(obj)

selected_galaxies = np.asarray(d['galaxies'])[GALAXY]

for gal in selected_galaxies:
    z = np.asarray(gal.mags[0].z)
    ind_z = np.argmin(abs(z - REDSHIFT))
    z = z[ind_z+1:][::-1]
    U = np.asarray(gal.mags[0].Abs[ind_z+1:][::-1])
    V = np.asarray(gal.mags[1].Abs[ind_z+1:][::-1])
    J = np.asarray(gal.mags[2].Abs[ind_z+1:][::-1])
    bhmass = np.zeros(len(z))
    for i in range(0,len(z)):
        bhmass[i] = float(gal.bhar[np.where(gal.z==z[i])]+1e-14)
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    x = U - V 
    y = V - J
    ax.plot(x,y, '-k')
    sc = ax.scatter(x,y,c=np.log10(bhmass),cmap='plasma',s=20)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(M_{BH}[M_{\odot}])$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_track_'+str(gal.progen_id)+'.png',format='png', dpi=250, bbox_inches='tight')