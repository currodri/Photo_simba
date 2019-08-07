
# Reads in the photometry files output by Loser (https://bitbucket.org/romeeld/closer/src/default/)
# These are ASCII files, with suffix .app for apparent mags, .abs for absolute mags.

#import pylab as plt
import numpy as np
import os
import caesar
import h5py

import sys
sys.path.insert(0, '../../SH_Project')
from galaxy_class import Magnitude

###########################################################################
def read_mags(infile,magcols, nodust=False):
    f = h5py.File(infile,'r')
    header = f['HEADER_INFO']
    redshift = float(header[0].split()[2])
    t_hubble = float(header[0].split()[6])
    caesar_id = f['CAESAR_ID'][:]
    colorinfo = f['COLOR_INFO'][:]
    Lapp = []
    Lapp_nd = []
    Labs = []
    Labs_nd = []
    filter_info = []
    for i in range(len(magcols)):
        imag = int(magcols[i])
        print ('Reading now filter for '+str(colorinfo[imag]))
        filter_info.append(str(colorinfo[imag]))
        Labs.append(f['absmag_%d'%imag])
        Lapp.append(f['appmag_%d'%imag])
        if nodust:
            Labs_nd.append(f['absmag_nodust_%d'%imag])
            Lapp_nd.append(f['appmag_nodust_%d'%imag])
    if nodust:   
        Labs = np.asarray(Labs)  # absolute magnitudes of galaxies in each desired band
        Labs_nd = np.asarray(Labs_nd)  # no-dust absolute magnitudes
        Lapp = np.asarray(Lapp)  # apparent magnitudes of galaxies in each desired band
        Lapp_nd = np.asarray(Lapp_nd)  # no-dust apparent magnitudes
        return redshift,t_hubble,filter_info,caesar_id,Labs,Labs_nd,Lapp,Lapp_nd
    else:
        Labs = np.asarray(Labs)  # absolute magnitudes of galaxies in each desired band
        Lapp = np.asarray(Lapp)  # apparent magnitudes of galaxies in each desired band
        return redshift,t_hubble,filter_info,caesar_id,Labs,Lapp

def crossmatch_loserandquench(MODEL,WIND,SNAP_0,galaxies,magcols):
    caesar_dir = '/home/rad/data/%s/%s/Groups/' % (MODEL,WIND)
    loser = filter(lambda file:file[-5:]=='.hdf5' and file[0]=='p' and int(file[-8:-5])<=SNAP_0, os.listdir('./'))
    loser_sorted = sorted(loser,key=lambda file: int(file[-8:-5]), reverse=True)

    for gal in galaxies:
        for i in range(0, len(magcols)):
            gal.mags.append(Magnitude())

    for l in range(0, len(loser_sorted)):
        redshift,t_hubble,filter_info,caesar_id,Labs,Lapp = read_mags(loser_sorted[l],magcols)
        print ('Reading loser file for z=%s' % (redshift))
        for i in range(0,len(caesar_id)):
            for gal in galaxies:
                for red_filt in gal.z[gal.caesar_id==float(caesar_id[i])]:
                    if red_filt == redshift:
                        for f in range(0, len(magcols)):
                            mag_data = gal.mags[f]
                            if l==0:
                                f_info = filter_info.split()
                                mag_data.filtername = f_info[6]+' '+f_info[7]+' '+f_info[8]
                                mag_data.wave_eff = float(f_info[5])
                            mag_data.z.append(redshift)
                            mag_data.Abs.append(Labs[f][l])
                            mag_data.App.append(Lapp[f][l])
    return galaxies

