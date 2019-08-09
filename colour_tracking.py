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
WIND = sys.argv[2]      # e.g. s50
SNAP = int(sys.argv[3])  # e.g. 0.5
GALAXY = [int(sys.argv[4])]    # e.g. 987
if len(sys.argv) > 5:
    GaLAXY = []
    for i in range(4,len(sys.argv)):
        GALAXY.append(int(sys.argv[i])) # this is for the case in which we want the tracks for multiple galaxies

# Read data from pickle file
data_file = '/home/curro/quenchingSIMBA/code/SH_Project/mandq_results_%s.pkl' % (MODEL)
obj = open(data_file, 'rb')
d = pickle.load(obj)

caesar_file = '/home/rad/data/%s/%s/Groups/%s_%s.hdf5' % (MODEL,WIND,MODEL,SNAP)
print(caesar_file)
cfile = h5py.File(caesar_file)
REDSHIFT = float(cfile['simulation_attributes'].attrs['redshift'])

selected_galaxies = np.asarray(d['galaxies'])[GALAXY]

x = []
y = []
for gala in d['galaxies']:
    mag_z = np.asarray(gala.mags[0].z)
    pos = np.where(mag_z==REDSHIFT)
    pos2 = np.where(gala.z==REDSHIFT)
    mag0 = np.asarray(gala.mags[0].Abs)
    if mag0[pos] and gala.t[0][pos2] and np.log10(gala.m[0][pos2])>=9.5:
        U = gala.mags[0].Abs[gala.mags[0].z==REDSHIFT]
        V = gala.mags[1].Abs[gala.mags[1].z==REDSHIFT]
        J = gala.mags[2].Abs[gala.mags[2].z==REDSHIFT] 
        x.append(V - J)
        y.append(U - V)
x_non = np.asarray(x)
y_non = np.asarray(y)
print(len(x_non),len(y_non))

markers = ['o','*','s','x']
m_sizes = [20,40]
props = dict(boxstyle='round', facecolor='white', edgecolor='k', alpha=0.7)

for gal in selected_galaxies:
    z = np.asarray(gal.mags[0].z)
    ind_z = np.argmin(abs(z - REDSHIFT))
    z = z[ind_z+1:][::-1]
    U = np.asarray(gal.mags[0].Abs[ind_z+1:][::-1])
    V = np.asarray(gal.mags[1].Abs[ind_z+1:][::-1])
    J = np.asarray(gal.mags[2].Abs[ind_z+1:][::-1])
    bhmass = np.zeros(len(z))
    bhar = np.zeros(len(z))
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    x = V - J
    y = U - V
    ax.plot(x,y, '-k')
    for i in range(0,len(z)):
        bhmass[i] = float(gal.bh_m[np.where(gal.z==z[i])])
        bhar[i] = float(gal.bhar[np.where(gal.z==z[i])])
    max_bhm = np.log10(np.amax(bhmass))
    max_bhar = np.amax(bhar)
    min_bhm = np.log10(np.amin(bhmass))
    min_bhar = np.amin(bhar)
    for i in range(0, len(z)):
        m = 0
        size = m_sizes[int(gal.g_type[np.where(gal.z==z[i])])]
        if gal.mergers:
            for merg in gal.mergers:
                if z[i]==gal.z[merg.indx]:
                    m = 1
                    ax.text(0.99*x[i], 1.05*y[i], r'$R = $'+'{:.3}'.format(merg.merger_ratio), fontsize=8, bbox=props)
        elif gal.quenching:
            for quench in gal.quenching:
                if z[i]==gal.z[quench.indx]:
                    m = 2
                    ax.text(0.99*x[i], 1.05*y[i], r'$\tau_{q} = $'+'{:.3}'.format(quench.quench_time)+r' Gyr', fontsize=8, bbox=props)
        elif gal.rejuvenations:
            for j in range(0, len(gal.rejuvenations)):
                if z[i]==gal.z[gal.rejuvenations[j]]:
                    m = 3
        sc = ax.scatter(x[i],y[i],c=bhmass[i],cmap='plasma',s=size, marker=markers[m])
        sc.set_clim(min_bhm,max_bhm)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(M_{BH}[M_{\odot}])$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_track_'+str(gal.progen_id)+'.png',format='png', dpi=250, bbox_inches='tight')