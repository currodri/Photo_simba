## IMPORT LIBRARIES
from astropy.io import fits
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

old_loser = h5py.File('../VWSC_simba/CATS/simba/loserpsb_m50n512_125.hdf5','r')
new_loser = h5py.File('/home/rad/data/m50n512/s50/Groups/pylosapp_m50n512_125.hdf5','r')
files = [old_loser,new_loser]

fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1,1,1)
ax.set_ylabel('Mag', fontsize=16)
ax.set_xlabel('Index', fontsize=16)
labels = ['Old loser', 'New loser']
for k in range(0, len(files)):
    f = files[k]
    caesar_id = f['iobjs'][:]
    print(len(caesar_id))
    if k==0:
        ind = [1,3,4]
    else:
        ind = [22,23,25]
    Lapp_old = np.zeros((len(caesar_id),len(ind)))
    #ind = [34,35,36,37,38,39,40,41,42,17,18]
    # Apparent magnitudes of galaxies in each desired band
    index = np.linspace(1,len(caesar_id),len(caesar_id))
    for i in range(0, len(ind)):
        Lapp_old[:,i] = np.sort(np.asarray(f['mymags'][:,ind[i]])) # Save mags for the selected filters
        ax.plot(index,Lapp_old[:,i],'-', label=labels[k])
ax.legend(loc='best', prop={'size': 12})
fig.savefig('catalogue_check.png',format='png', dpi=250, bbox_inches='tight')