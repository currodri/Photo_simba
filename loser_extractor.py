
# Reads in the photometry files output by Loser (https://bitbucket.org/romeeld/closer/src/default/)
# These are ASCII files, with suffix .app for apparent mags, .abs for absolute mags.

import numpy as np
import os
import h5py
from astropy.cosmology import FlatLambdaCDM
import sys
sys.path.insert(0, '../../SH_Project')
from galaxy_class import Magnitude

###########################################################################
def read_mags(infile,magcols,MODEL,WIND,nodust=False):
    f = h5py.File(infile,'r')
    snap = infile[-8:-5]
    caesar_file = '/home/rad/data/%s/%s/Groups/%s_%s.hdf5' % (MODEL,WIND,MODEL,snap)
    print(caesar_file)
    cfile = h5py.File(caesar_file)
    ngal = len(cfile['galaxy_data']['GroupID'])
    redshift = cfile['simulation_attributes'].attrs['redshift']
    omega_matter = cfile['simulation_attributes'].attrs['omega_matter']
    omega_baryon = cfile['simulation_attributes'].attrs['omega_baryon']
    h = cfile['simulation_attributes'].attrs['hubble_constant']
    cosmo = FlatLambdaCDM(H0=100*h, Om0=omega_matter, Ob0=omega_baryon,Tcmb0=2.73)  # set our cosmological parameters
    t_hubble = cosmo.age(redshift).value
    caesar_id = f['CAESAR_ID'][:]
    colorinfo = f['COLOR_INFO'][:]
    Lapp = []
    Lapp_nd = []
    Labs = []
    Labs_nd = []
    filter_info = []
    for i in range(len(magcols)):
        imag = int(magcols[i])
        filter_info.append(colorinfo[imag])
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
        return ngal,redshift,t_hubble,filter_info,caesar_id,Labs,Lapp

def crossmatch_loserandquench(MODEL,WIND,SNAP_0,galaxies,magcols):
    caesar_dir = '/home/rad/data/%s/%s/Groups/' % (MODEL,WIND)
    loser = filter(lambda file:file[-5:]=='.hdf5' and file[0]=='p' and int(file[-8:-5])<=SNAP_0, os.listdir(caesar_dir))
    loser_sorted = sorted(loser,key=lambda file: int(file[-8:-5]), reverse=True)

    for i in range(0,len(galaxies)):
        for j in range(0, len(magcols)):
            galaxies[i].mags.append(Magnitude())
    print(loser_sorted)
    for l in range(0, len(loser_sorted)):
        print(loser_sorted)
        print(loser_sorted[l])
        ngal,redshift,t_hubble,filter_info,caesar_id,Labs,Lapp = read_mags(caesar_dir+loser_sorted[l],magcols,MODEL,WIND)
        print ('Reading loser file for z=%s' % (redshift))
        counter = 0
        for gal in galaxies:
            if gal.caesar_id[gal.z==float(redshift)]:
                indx = int(gal.caesar_id[gal.z==float(redshift)])
                for f in range(0, len(magcols)):
                    if l==0:
                        f_info = filter_info[f].split()
                        gal.mags[f].filtername = f_info[6]+' '+f_info[7]+' '+f_info[8]
                        gal.mags[f].wave_eff = float(f_info[5])
                    gal.mags[f].z.append(redshift)
                    gal.mags[f].Abs.append(Labs[f][indx])
                    gal.mags[f].App.append(Lapp[f][indx])
                counter = counter + 1
        print('Cross-matched '+str(counter)+' galaxies out of '+str(ngal))
    return galaxies

