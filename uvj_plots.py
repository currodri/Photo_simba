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
    h2_change = []
    h1_change = []
    bhar_change = []
    bhm_change = []
    tbt = []
    U_non = []
    V_non = []
    J_non = []
    print('Possible PSBs found for galaxies: ')
    for gal in galaxies:
        mag_z = np.asarray(gal.mags[0].z)
        pos = np.where(mag_z==redshift)
        pos2 = np.where(gal.z==redshift)
        mag0 = np.asarray(gal.mags[0].Abs)
        if mag0[pos] and gal.t[0][pos2]:
            if gal.quenching and not isinstance(gal.t[1],int):
                possible_q = []
                possible_tau = []
                possible_ssfr = []
                possible_h2 = []
                possible_h1 = []
                possible_bhar = []
                possible_bhm = []
                possible_tbt = [] 
                for quench in gal.quenching:
                    start = quench.above9
                    end = quench.below11
                    ssfr = float(gal.ssfr[0][pos2])
                    snap_t = gal.t[0][pos2]
                    ssfr_cond = 10**sfr_condition('end', snap_t)
                    if gal.t[1][end] <= snap_t and (snap_t-gal.t[1][end]) <= 1.0 and ssfr <= ssfr_cond:
                        possible_q.append(snap_t-gal.t[1][end])
                        possible_tau.append(quench.quench_time/gal.t[1][end])
                        possible_ssfr.append(ssfr)

                        diff = abs(gal.t[0] - gal.t[1][start])
                        ind_start = np.argmin(diff)
                        h2_s = gal.h2_gas[ind_start]/gal.m[0][ind_start] + 1e-2
                        h2_e = gal.h2_gas[quench.indx]/gal.m[0][quench.indx] + 1e-2
                        possible_h2.append(np.log10(h2_e/h2_s))
                        h1_s = gal.h1_gas[ind_start]/gal.m[0][ind_start] + 1e-2
                        h1_e = gal.h1_gas[quench.indx]/gal.m[0][quench.indx] + 1e-2
                        possible_h1.append(np.log10(h1_e/h1_s))
                        bhar_s = gal.bhar[ind_start] + 1e-3
                        bhar_e = gal.bhar[quench.indx] + 1e-3
                        bhm_s = gal.bh_m[ind_start] + 1e+7
                        bhm_e = gal.bh_m[quench.indx] + 1e+7
                        possible_bhm.append(np.log10(bhm_e/bhm_s))
                        possible_bhar.append(np.log10(bhar_e/bhar_s))
                        possible_tbt.append('b')
                        for i in range(int(pos2[0]),len(gal.t[0])):
                            ssfr_cond = 10**sfr_condition('end',gal.t[0][i])
                            if gal.ssfr[0][i] >= ssfr_cond:
                                possible_tbt[-1] = 'r'

                if possible_q:
                    possible_q = np.asarray(possible_q)
                    possible_tau = np.asarray(possible_tau)
                    possible_ssfr = np.asarray(possible_ssfr)
                    mag1 = np.asarray(gal.mags[1].Abs)
                    mag2 = np.asarray(gal.mags[2].Abs)
                    U.append(gal.mags[0].Abs[gal.mags[0].z==redshift])
                    V.append(gal.mags[1].Abs[gal.mags[1].z==redshift])
                    J.append(gal.mags[2].Abs[gal.mags[2].z==redshift])
                    sSFR.append(possible_ssfr[np.argmin(possible_q)])
                    q_time.append(np.amin(possible_q))
                    tau_q.append(possible_tau[np.argmin(possible_q)])
                    h2_change.append(possible_h2[np.argmin(possible_q)])
                    h1_change.append(possible_h1[np.argmin(possible_q)])
                    bhar_change.append(possible_bhar[np.argmin(possible_q)])
                    bhm_change.append(possible_bhm[np.argmin(possible_q)])
                    tbt.append(possible_tbt[np.argmin(possible_q)])
                    if 0.7 <= (V[-1]-J[-1]) < 0.8 and 1.5 <= (U[-1]-V[-1]) < 1.63:
                        print('Caesar ID: '+str(gal.caesar_id[pos2]), 'Progen ID: '+str(gal.progen_id))
                else:
                    U_non.append(gal.mags[0].Abs[gal.mags[0].z==redshift])
                    V_non.append(gal.mags[1].Abs[gal.mags[1].z==redshift])
                    J_non.append(gal.mags[2].Abs[gal.mags[2].z==redshift])
            elif np.log10(gal.m[0][pos2]) >= masslimit:
                U_non.append(gal.mags[0].Abs[gal.mags[0].z==redshift])
                V_non.append(gal.mags[1].Abs[gal.mags[1].z==redshift])
                J_non.append(gal.mags[2].Abs[gal.mags[2].z==redshift])
    q_time = np.asarray(q_time)
    tau_q = np.asarray(tau_q)
    sSFR = np.asarray(sSFR)
    h2_change = np.asarray(h2_change)
    h1_change = np.asarray(h1_change)
    bhar_change = np.asarray(bhar_change)
    bhm_change = np.asarray(bhm_change)
    tbt = np.asarray(tbt)
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
    sc = ax.scatter(x,y,c=np.log10(sSFR+1e-14),cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(sSFR)$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qssfr_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=h2_change,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\Delta f_{H2})$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qh2_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=h1_change,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\Delta f_{H1})$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qh1_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=bhar_change,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\Delta BHAR)$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qbhar_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=bhm_change,cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\Delta M_{BH})$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qbhm_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    ax.scatter(x,y,c=tbt, s=8)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qtbt_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    
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
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    sc = ax.scatter(x,y,c=np.log10(tau_q),cmap='plasma',s=8)
    cb = fig2.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(\tau_q/t_{H})$', fontsize=16)
    fig2.tight_layout()
    fig2.savefig('../color_plots/uv_vj_qscale_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    

    fig3 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig3.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    x = ma.array(x)
    y = ma.array(y)
    tau_q = ma.array(tau_q)
    x_fast = ma.masked_where(tau_q>=10**(-1.5),x)
    y_fast = ma.masked_where(tau_q>=10**(-1.5),y)
    x_slow = ma.masked_where(tau_q<10**(-1.5),x)
    y_slow = ma.masked_where(tau_q<10**(-1.5),y)
    ax.scatter(x_slow,y_slow,s=8, c = 'b', label='Slow quenching')
    ax.scatter(x_fast,y_fast,s=8, c = 'r', label='Fast quenching')
    ax.legend(loc='best')
    fig3.tight_layout()
    fig3.savefig('../color_plots/uv_vj_qsf_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')


def uvj_plot(redshift,galaxies,masslimit):

    x = []
    y = []
    counter = 0
    for gal in galaxies:
        mag_z = np.asarray(gal.mags[0].z)
        pos = np.where(mag_z==redshift)
        pos2 = np.where(gal.z==redshift)
        mag0 = np.asarray(gal.mags[0].Abs)
        if mag0[pos] and gal.t[0][pos2] and np.log10(gal.m[0][pos2])>=masslimit:
            U = gal.mags[0].Abs[gal.mags[0].z==redshift]
            V = gal.mags[1].Abs[gal.mags[1].z==redshift]
            J = gal.mags[2].Abs[gal.mags[2].z==redshift] 
            x.append(V - J)
            y.append(U - V)
            counter = counter + 1
    print(counter)
    x = np.asarray(x)
    y = np.asarray(y)
    fig3 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig3.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x, y, gridsize=50,bins='log', cmap='Greys')
    fig3.tight_layout()
    fig3.savefig('../color_plots/uv_vj_hexbin_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')

uvj_quench(REDSHIFT,d['galaxies'],MASSLIMIT)

#uvj_plot(REDSHIFT,d['galaxies'],MASSLIMIT)
