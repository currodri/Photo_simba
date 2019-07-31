
# Reads in the photometry files output by Loser (https://bitbucket.org/romeeld/closer/src/default/)
# These are ASCII files, with suffix .app for apparent mags, .abs for absolute mags.

#import pylab as plt
import numpy as np
import sys
import os
import function as fu
import caesar
import h5py
import cPickle as pickle

import sys
sys.path.insert(0, '../../SH_Project')
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from astropy.cosmology import FlatLambdaCDM
# choose simulation and snapshot to look at
MODEL = sys.argv[1]  # e.g. m50n512
WIND = sys.argv[2]  # e.g. s50 for Simba
SNAP = sys.argv[3]  # snapshot number
magcols = [0]  # default colors
if len(sys.argv) > 4:
    magcols = sys.argv[4:] # the desired color numbers from the header of the *.app or *.abs files.  e.g. 0 for Johnson V, and so on.

###########################################################################
def sfr_condition_2(type, time):
    if type == 'start':
        lsfr = np.log10(1/(time))-9
    elif type == 'end':
        lsfr  = np.log10(0.2/(time))-9
    return lsfr
def read_mags(infile,magcols):
    f = h5py.File(infile,'r')
    header = f['HEADER_INFO']
    redshift = float(header[0].split()[2])
    t_hubble = float(header[0].split()[6])
    boxsize = (header[0].split()[10])
    sfr = f['SFR'][:]
    mstar = f['MSTAR'][:]
    LyC = f['LyC'][:]
    mformed = f['MFORMED'][:]
    L_FIR = f['L_FIR'][:]
    age = f['AGE'][:]
    Zstar = f['ZSTAR'][:]
    A_V = f['A_V'][:]
    nbands = len(f['COLOR_INFO'])
    caesar_id = f['CAESAR_ID'][:]
    colorinfo = f['COLOR_INFO'][:]
    print(colorinfo)
    ngal = len(sfr)
    Lapp = []
    Lapp_nd = []
    Labs = []
    Labs_nd = []
    for i in range(len(magcols)):
        imag = int(magcols[i])
        print ('Reading now filter for '+str(colorinfo[imag]))
        Labs.append(f['absmag_%d'%imag])
        Labs_nd.append(f['absmag_nodust_%d'%imag])
        Lapp.append(f['appmag_%d'%imag])
        Lapp_nd.append(f['appmag_nodust_%d'%imag])
    Labs = np.asarray(Labs)  # absolute magnitudes of galaxies in each desired band
    Labs_nd = np.asarray(Labs_nd)  # no-dust absolute magnitudes
    Lapp = np.asarray(Lapp)  # apparent magnitudes of galaxies in each desired band
    Lapp_nd = np.asarray(Lapp_nd)  # no-dust apparent magnitudes
    return redshift,t_hubble,boxsize,colorinfo,nbands,ngal,caesar_id,sfr,LyC,mformed,mstar,L_FIR,age,Zstar,A_V,Labs,Labs_nd,Lapp,Lapp_nd

#def SED_plotting(infile, galaxy_id)

def read_caesar(caesarfile):
    # example of how to read caesar catalog file.  this one only reads in M* and SFR, just
    # to cross-check with Loser file, but Caesar catalogs contain a huge amount of other info
    sim = caesar.load(caesarfile,LoadHalo=False)
    z = sim.simulation.redshift
    ms = np.asarray([i.masses['stellar'] for i in sim.galaxies])
    sfr = np.asarray([i.sfr for i in sim.galaxies])
    return z,ms,sfr

def uv_vj_plot(ngal, Lapp, SFR=0, MS=0):
    # Simple function that provides U-V vs V_J colours for all the galaxies in a given snapshot
    # If SFR and stellar mass are provided, scatter points are color-coded with sSFR
    # The bands in Lapp should be given as Lapp[0] = U, Lapp[1] = V and Lapp[2] = J
    y = np.zeros(ngal) # U-V
    x = np.zeros(ngal) # V-J
    for i in range(0, ngal):
        y[i] = Lapp[0][i] - Lapp[1][i]
        x[i] = Lapp[1][i] - Lapp[2][i]
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    if isinstance(SFR, np.ndarray) and isinstance(MS, np.ndarray):
        sSFR = SFR/MS + 1e-14
        sc = ax.scatter(x,y,c=np.log10(sSFR),cmap='plasma',s=8)
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
        cb.set_label(label=r'$\log$(sSFR[yr$^{-1})$', fontsize=16)
        fig.tight_layout()
        fig.savefig('./color_plots/uv_vj_ssfr_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    else:
        ax.hexbin(x, y, gridsize=50,bins='log', cmap='Greys')
        fig.tight_layout()
        fig.savefig('../color_plots/uv_vj_hexbin_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
def histo_mag(ngal, Lapp, filtername, nbins):
    # Simple function that provides the histogram distribution for a given band.
    # It requires the magnitudes in that band Lapp, the name of the filter from 
    # the loser colorinfo and the number of bins.

    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(str(filtername[0])+str(filtername[1])+'Magnitude', fontsize=16)
    ax.set_ylabel(r'$N/N_{Tot}$', fontsize=16)
    bin_count, bin_edges = np.histogram(Lapp, bins=nbins)
    bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])
    ax.step(bin_cent, bin_count/float(ngal), where='mid')
    fig.tight_layout()
    fig.savefig('../color_plots/'+str(filtername[0])+str(filtername[1])+'_histo_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
def scatter_app_vs_mass(ngal,Lapp,mass,filtername):
    # This plots the scattered distribution for a given band wrt the stellar mass 
    fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel(str(filtername[0])+str(filtername[1])+'Magnitude', fontsize=16)
    ax.set_xlabel(r'$\log(M_*[M_{\odot}])$', fontsize=16)
    ax.hexbin(mass, Lapp, gridsize=50,bins='log', cmap='Greys')
    fig.tight_layout()
    fig.savefig('../color_plots/'+str(filtername[0])+str(filtername[1])+'_scatter_mass_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight') 
def uvj_mergertime(redshift,caesar_id,Labs, merger_file):
    # This function obtains the UVJ colour plot for the galaxies that a given redshift experienced a merger
    # as determined by the mergerFinder algorithm. The scatter points are colour coded with the time past 
    # after the last merger.
    redshift_2 = 0.4904351460029006
    cosmo = FlatLambdaCDM(H0=100*0.68, Om0=0.3, Ob0=0.04799952624117699,Tcmb0=2.73)
    t_hubble = cosmo.age(redshift_2).value
    # Start by selecting the galaxies in the snapshot that experienced a merger before
    file = open(merger_file, 'rb')
    d = pickle.load(file)
    print('Merger pickle file read.')
    mergers = d['mergers']
    U = []
    V = []
    J = []
    m_time = []
    U_non = []
    V_non = []
    J_non = []
    for i in range(0,len(caesar_id)):
        #print('caesar_id',caesar_id[i])
        possible_m = []
        for merg in mergers:
            for red_filt in merg.all_z[merg.caesar_id==float(caesar_id[i])]:
                if red_filt == redshift_2 and merg.galaxy_t[2] <= t_hubble:
                    diff_t = t_hubble - merg.galaxy_t[2]
                    possible_m.append(diff_t)
        if possible_m:
            possible_m = np.asarray(possible_m)
            m_time.append(np.amin(possible_m))
            U.append(Labs[0][i])
            V.append(Labs[1][i])
            J.append(Labs[2][i])
        else:
            U_non.append(Labs[0][i])
            V_non.append(Labs[1][i])
            J_non.append(Labs[2][i])
    m_time = np.asarray(m_time)
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
    sc = ax.scatter(x,y,c=np.log10(m_time),cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(t_h - t_m)$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_mtime_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    print('Merger plot done.')

def uvj_quench(redshift,caesar_id,Labs,sfr,mstar,quench_file):
    # This function obtains the UVJ colour plot for the galaxies that a given redshift experienced a quenching
    # as determined by the quenchingFinder algorithm. The scatter points are colour coded with the time past 
    # after the last quenching (Fig 1) or the quenching timescale (Fig 2).
    redshift_2 = 0.4904351460029006
    cosmo = FlatLambdaCDM(H0=100*0.68, Om0=0.3, Ob0=0.04799952624117699,Tcmb0=2.73)
    t_hubble = cosmo.age(redshift_2).value
    # Start by selecting the galaxies in the snapshot that experienced a merger before
    file = open(quench_file, 'rb')
    d = pickle.load(file)
    quenchings = d['quenched_galaxies']
    print('Quench pickle file read.')
    U = []
    V = []
    J = []
    q_time = []
    sSFR = []
    tau_q = []
    U_non = []
    V_non = []
    J_non = []
    U_fast = []
    V_fast = []
    J_fast = []
    U_slow = []
    V_slow = []
    J_slow = []
    for i in range(0,len(caesar_id)):
        #print('caesar id', caesar_id[i])
        possible_q = []
        possible_tau = []
        for j in range(0, len(quenchings)):
            galaxy = quenchings[j]
            for red_filt in galaxy.all_z[galaxy.caesar_id==float(caesar_id[i])]:
                if red_filt == redshift_2:
                   # print('Match found')
                    for quench in galaxy.quenching:
                        end = quench.below11
                        if galaxy.galaxy_t[end] <= t_hubble:
                            for k in range(0, len(galaxy.rate), 3):
                                if galaxy.galaxy_t[end] <= galaxy.rate[k+1] < t_hubble and np.log10(sfr[i]/(10**mstar[i])+1e-14) >= sfr_condition_2('start',t_hubble):
                                    print(galaxy.galaxy_t[end],galaxy.rate[k+1])
                                    print(np.log10(sfr[i]/(10**mstar[i])+1e-14), sfr_condition_2('start',t_hubble))
                                    pass
                                else:
                                    possible_q.append(galaxy.galaxy_t[end])
                                    possible_tau.append(quench.quench_time/galaxy.galaxy_t[end])
        if possible_q:
            possible_q = np.asarray(possible_q)
            diff = t_hubble - possible_q
            q_time.append(np.amin(diff))
            tau_q.append(possible_tau[np.argmin(diff)])
            U.append(Labs[0][i])
            V.append(Labs[1][i])
            J.append(Labs[2][i])
            sSFR.append(np.log10(sfr[i]/(10**mstar[i])+1e-14))
            if q_time[-1] > 1 and (U[-1]-V[-1]) < 1.5 and (V[-1]-J[-1]) < 1.0:
                print(np.log10(sfr[i]/(10**mstar[i])+1e-14),sfr_condition_2('end',t_hubble))
            if possible_tau[np.argmin(diff)] >= (10**(-1.5)):
                U_slow.append(Labs[0][i])
                V_slow.append(Labs[1][i])
                J_slow.append(Labs[2][i]) 
            else:
                U_fast.append(Labs[0][i])
                V_fast.append(Labs[1][i])
                J_fast.append(Labs[2][i]) 
        else:
            U_non.append(Labs[0][i])
            V_non.append(Labs[1][i])
            J_non.append(Labs[2][i])
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
    sc = ax.scatter(x,y,c=np.log10(q_time),cmap='plasma',s=8)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(label=r'$\log(t_h - t_q)$', fontsize=16)
    fig.tight_layout()
    fig.savefig('../color_plots/uv_vj_qtime_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')
    print('First quench plot done')
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
    
    U_slow = np.asarray(U_slow)
    V_slow = np.asarray(V_slow)
    J_slow = np.asarray(J_slow)
    x_slow = V_slow - J_slow
    y_slow = U_slow - V_slow
    U_fast = np.asarray(U_fast)
    V_fast = np.asarray(V_fast)
    J_fast = np.asarray(J_fast)
    x_fast = V_fast - J_fast
    y_fast = U_fast - V_fast
    fig3 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig3.add_subplot(1,1,1)
    ax.set_xlabel('V - J', fontsize=16)
    ax.set_ylabel('U - V', fontsize=16)
    ax.hexbin(x_non, y_non, gridsize=50,bins='log', cmap='Greys')
    ax.scatter(x_slow,y_slow,s=8, c = 'b', label='Slow quenching')
    ax.scatter(x_fast,y_fast,s=8, c = 'r', label='Fast quenching')
    ax.legend(loc='best')
    #cb = fig2.colorbar(sc, ax=ax, orientation='horizontal')
    #cb.set_label(label=r'$\log(\tau_q/t_{H})$', fontsize=16)
    fig3.tight_layout()
    fig3.savefig('../color_plots/uv_vj_qsf_'+str(SNAP)+'.png',format='png', dpi=250, bbox_inches='tight')


###########################################################################

if __name__ == '__main__':

    lfile = '/home/rad/data/%s/%s/Groups/phot_%s_%03d.hdf5' % (MODEL,WIND,MODEL,int(SNAP))
    redshift,t_hubble,boxsize,colorinfo,nbands,ngal,caesar_id,sfr,LyC,mformed,mstar,L_FIR,meanage,Zstar,A_V,Labs,Labs_nd,Lapp,Lapp_nd = read_mags(lfile,magcols)
    print ('z=',redshift,'L=',boxsize,'Nbands=',nbands,'Ngal=',ngal) #,'\n',mstar,'\n',Lmag,'\n',Lmag_nd
    cfile = '/home/rad/data/%s/%s/Groups/%s_%03d.hdf5' % (MODEL,WIND,MODEL,int(SNAP))
    #redshift_caesar,ms_caesar,sfr_caesar = read_caesar(cfile)
    #print(sfr)
    #print(mstar)
    # check that the caesar and phot files indeed have the same M*,SFRs, to within tol
    #tol = 0.001
    #for i in range(len(ms_caesar)):
        #if mstar[i] < 5.8e8: continue
        #if abs(ms_caesar[i]-mstar[i])>tol*ms_caesar[i] or abs(sfr_caesar[i]-sfr[i])>tol*sfr_caesar[i]:
            #print 'trouble',i,np.log10(ms_caesar[i]),np.log10(mstar[i]),sfr_caesar[i],sfr[i]
    #print(Labs[0][0:20])
    #print(Lapp[0][0]-Labs[0][0])
    #print(Lapp[1][20]-Labs[1][20])
    #print(Lapp[2][10]-Labs[2][10])
    #filtername = colorinfo[9].split()[6:8]
   # uv_vj_plot(ngal,Lapp)#,SFR=sfr,MS=mstar)
    #histo_mag(ngal, Lapp[0],filtername, 10)
    #scatter_app_vs_mass(ngal, Lapp[0], mstar, filtername)
    print('Now starting to make plots...')
    merg_data = '/home/curro/quenchingSIMBA/code/mergers/%s/merger_results.pkl' % (MODEL)
    quench_data = '/home/curro/quenchingSIMBA/code/quench_analysis/%s/quenching_results.pkl' % (MODEL)
    #uvj_mergertime(redshift,caesar_id,Labs,merg_data)
    uvj_quench(redshift,caesar_id,Labs,sfr,mstar,quench_data)
