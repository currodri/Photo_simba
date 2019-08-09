#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09 August 2019

@author: Curro Rodriguez Montero, School of Physics and Astronomy,
            University of Edinburgh, JCMB, King's Buildings

This code provides a way to make UVJ plots that combine the results of the mergerFinder and quenchingFinder algorithms with
the photometry data of cLoser (https://bitbucket.org/romeeld/closer/src/default/).

For questions about the code:
s1650043@ed.ac.uk
"""
# Import necessary libraries
import numpy as np
from numpy import ma
import cPickle as pickle 
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import sys
sys.path.insert(0, '../../SH_Project')
from galaxy_class import Magnitude, GalaxyData

# Get details from terminal
MODEL = sys.argv[1]     # e.g. m100n1024
WIND = sys.argv[2]      # e.g. s50
SNAP = int(sys.argv[3]) # e.g. 0.5
MASSLIMIT = float(sys.argv[4]) # usually it should be log(M*)>9.5

caesar_file = '/home/rad/data/%s/%s/Groups/%s_%s.hdf5' % (MODEL,WIND,MODEL,SNAP)
print(caesar_file)
cfile = h5py.File(caesar_file)
REDSHIFT = float(cfile['simulation_attributes'].attrs['redshift'])
print('Making UVJ plots for z = '+str(REDSHIFT))

# Read data from pickle file
data_file = '/home/curro/quenchingSIMBA/code/SH_Project/mandq_results_%s.pkl' % (MODEL)
obj = open(data_file, 'rb')
d = pickle.load(obj)

def sfr_condition(type, time):
    if type == 'start':
        lsfr = np.log10(1/(time))-9
    elif type == 'end':
        lsfr  = np.log10(0.2/(time))-9
    return lsfr

# Plotting routines
def uvj_quench(redshift,galaxies,masslimit):
    # This function obtains the UVJ colour plot for the galaxies that a given redshift experienced a quenching
    # as determined by the quenchingFinder algorithm. The scatter points are colour coded with the time past 
    # after the last quenching (Fig 1) or the quenching timescale (Fig 2).

    # Start by selecting the galaxies in the snapshot that experienced a quenching before and are still quenched
    U = []
    V = []
    J = []
    q_time = []
    sSFR = []
    tau_q = []
    U_non = []
    V_non = []
    J_non = []
    for gal in galaxies:
        mag_z = np.asarray(gal.mags[0].z)
        pos = np.where(mag_z==redshift)
        pos2 = np.where(gal.z==redshift)
        mag0 = np.asarray(gal.mags[0].Abs)
        if mag0[pos] and gal.t[0][pos2]:
            m_gal = np.asarray(gal.m)
            if gal.quenching:
                possible_q = []
                possible_tau = []
                for quench in gal.quenching:
                    indx = quench.indx
                    ssfr = gal.ssfr[0][pos2]
                    snap_t = gal.t[0][pos2]
                    ssfr_cond = 10**sfr_condition('end', snap_t)
                    if gal.t[0][indx] <= snap_t and (snap_t-gal.t[0][indx]) <= 1.0 and ssfr <= ssfr_cond:
                        mag1 = np.asarray(gal.mags[1].Abs)
                        mag2 = np.asarray(gal.mags[2].Abs)
                        U.append(mag0[pos])
                        V.append(mag1[pos])
                        J.append(mag2[pos])
                        sSFR.append(ssfr)
                        q_time.append(snap_t-gal.t[0][indx])
                        tau_q.append(quench.quench_time)
            elif np.log10(m_gal[0][pos2]) >= masslimit:
                U_non.append(gal.mags[0].Abs[gal.mags[0].z==redshift])
                V_non.append(gal.mags[1].Abs[gal.mags[1].z==redshift])
                J_non.append(gal.mags[2].Abs[gal.mags[2].z==redshift])
    q_time = np.asarray(q_time)
    tau_q = np.asarray(tau_q)
    U = np.asarray(U)
    V = np.asarray(V)
    J = np.asarray(J)
    x = V - J
    y = U - V
    U_non = np.asarray(U_non)
    V_non = np.asarray(V_non)
    J_non = np.asarray(J_non)
    x_non = V_non - J_non
    y_non = U_non - V_non

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=sSFR,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(sSFR)$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qssfr_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=q_time,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$t_h - t_q$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qtime_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
   
    fig2 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig2.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    print(len(tau_q), len(x), len(y))
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=np.log10(tau_q),cmap='plasma',s=8)
    cb = fig2.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\tau_q/t_{H})$', fontsize=16)
    fig2.tight_layout()
    fig2.savefig('../color_plots/uv_vj_qscale_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    

    x_slow = ma.masked_values(x, tau_q>=10**(-1.5))
    y_slow = ma.masked_values(y, tau_q>=10**(-1.5))
    x_fast = ma.masked_values(x, tau_q<10**(-1.5))
    y_fast = ma.masked_values(y, tau_q<10**(-1.5))
    fig3 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig3.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    ax.scatter(x_slow,y_slow,s=8, c = 'b', label='Slow quenching')
    ax.scatter(x_fast,y_fast,s=8, c = 'r', label='Fast quenching')
    ax.legend(loc='best')
    fig3.tight_layout()
    fig3.savefig('../color_plots/uv_vj_qsf_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')


uvj_quench(REDSHIFT,d['galaxies'],MASSLIMIT)