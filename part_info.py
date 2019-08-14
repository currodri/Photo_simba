#-----------------------------------------------------------------------------------
# PARTICLE INFO EXTRACTION CODE vs1.0
#
# These routines obtain specific data from sta, gas and black hole particles for the
# the galaxies in a given snapshot -- all of interest for the study of mergers and 
# quenching.

# Given a particular MODEL, WIND and SNAP and a set of galaxies IDs, it provides for
# each galaxy:
# - SFR per gas particle (M_sun yr^-1)
# - Total metal mass fraction per particle of gas and star
# - Z fraction for gas and stars for the elements chosen in terminal
# - Mass of star and gas particles
# - Age of star particle (Gyr)
# - Mass and accretion rate of most massive black hole


import caesar
from readgadget import *
import sys
import numpy as np
from yt.utilities.cosmology import Cosmology
from yt.units import km, s, Mpc
import cPickle as pickle
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

# FYI: Gizmo metallicity structure
def AbundanceLookup(MetType):
    MetName = ['Total','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
    SolarAbundances=[0.0134, 0.2485, 2.38e-3, 0.70e-3, 5.79e-3, 1.26e-3, 7.14e-4, 6.17e-4, 3.12e-4, 0.65e-4, 1.31e-3]
    return MetName[MetType],SolarAbundances[MetType]

# Age of universe at expansion factor a, in LCDM model
def a_to_t(a,omega_m,H0):
    theta = np.arctan(np.sqrt(omega_m/(1.-omega_m)) * np.power(a,-1.5))
    time = 2./(3*H0*np.sqrt(1.-omega_m)) * np.log((1+np.cos(theta))/np.sin(theta))
    return time.to('Gyr')

# read input files from command line
def obtain_snap_shell():
    MODEL = sys.argv[1]
    WIND = sys.argv[2]
    SNAP = int(sys.argv[3])
    snapfile = '/home/rad/data/%s/%s/snap_%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)
    caesarfile = '/home/rad/data/%s/%s/Groups/%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)
    # C and O are the two default metals
    metal1 = 2      # 2=C
    metal2 = 4      # 4=O
    if len(sys.argv)>4:
        metal1 = int(sys.argv[3])
        metal2 = int(sys.argv[4])
    # Setting some particular parameters
    ageyoung = 0.01  # "young" stars have age less than this in Gyr
    ageold = 1.0  # "old" stars have age less than this in Gyr
    Rcent = 1.0  # radius of central region in ckpc
    return snapfile, caesarfile, metal1, metal2
# load caesar file and provide basic cosmological parameters
def read_caesar(c_file,selected_gals):
    print('Reading caesar file...')
    sim = caesar.load(c_file,LoadHalo=False)
    redshift = np.round(sim.simulation.redshift,decimals=2)
    h = sim.simulation.hubble_constant
    cosmo = Cosmology(hubble_constant=sim.simulation.hubble_constant, omega_matter=sim.simulation.omega_matter, omega_lambda=sim.simulation.omega_lambda, omega_curvature=0)
    thubble = cosmo.hubble_time(redshift).in_units("Gyr")
    H0 = (100*sim.simulation.hubble_constant) * km / s / Mpc
    rhocrit = 3.*H0.to('1/s')**2 / (8*np.pi*sim.simulation.G)
    mlim = 32*rhocrit.to('Msun/kpc**3')*sim.simulation.boxsize.to('kpc')**3*sim.simulation.omega_baryon/sim.simulation.effective_resolution**3/sim.simulation.scale_factor**3 # galaxy mass resolution limit: 32 gas particle masses
    
    gals = np.array([])
    # read galaxy particle data for the selected galaxies
    if isinstance(selected_gals, np.ndarray):
        gals = np.asarray([i for i in sim.galaxies if i.GroupID in selected_gals and i.masses['stellar']>mlim])   # select resolved galaxies
    print('Galaxy data from caesar file extracted and saved.')
    return sim, redshift, h, cosmo, thubble, H0, rhocrit, mlim, gals

# load snap file and obtain speccific info for stellar, gas and bh particles
def read_snap(snapfile,sim,metals):
    print('Reading snapshot file...')
    print(snapfile)
    # read in the particle information from the snapshot
    smass = readsnap(snapfile,'mass','star',units=1,suppress=1)/h  # star particle mass Mo; note the h^-1 from Gadget units
    smetarray = readsnap(snapfile,'Metallicity','star',units=1,suppress=1)  # star metal array (metal mass fractions)
    sage = readsnap(snapfile,'age','star',units=1,suppress=1)   # expansion factor at time of formation
    sage = a_to_t(sage,sim.simulation.omega_matter,H0)  # age in Gyr

    gmass = readsnap(snapfile,'mass','gas',units=1,suppress=1)/h  # gas particle mass Mo; note the h^-1 from Gadget units
    gsfr = readsnap(snapfile,'sfr','gas',units=1,suppress=1)  # gas particle SFR Mo/yr
    gmetarray = readsnap(snapfile,'Metallicity','gas',units=1,suppress=1)  # gas metal array
    #gtemp = readsnap(snapfile,'u','gas',units=1,suppress=1)/h  # gas temp in K
    #gnh = readsnap(snapfile,'rho','gas',units=1,suppress=1)*h*h*0.76/1.673e-24  # number density in H atoms/cm^3
    gtemp = np.array([])
    gnh = np.array([])
    gpos = readsnap(snapfile,'pos','gas',units=1,suppress=1)/h  # gas pos in ckpc

    bhmass = readsnap(snapfile, 'BH_Mass','bndry',suppress=1)*1.e10/h  # in Mo
    bhmdot = readsnap(snapfile, 'BH_Mdot','bndry',suppress=1)*1.e10/h/3.08568e+16*3.155e7  # in Mo/yr

    # look up desired metallicities for gas and stars
    Zname1,abund1 = AbundanceLookup(metal1)
    Zname2,abund2 = AbundanceLookup(metal2)
    pZstar = np.asarray([i[0] for i in smetarray])  # total metal mass frac
    pCstar = np.asarray([i[metals[0]] for i in smetarray])  # star carbon metal frac
    pOstar = np.asarray([i[metals[1]] for i in smetarray])  # star oxygen metal frac
    pZgas = np.asarray([i[0] for i in gmetarray])
    pCgas = np.asarray([i[metals[0]] for i in gmetarray])
    pOgas = np.asarray([i[metals[1]] for i in gmetarray])
    print('Data from stellar, gas and bh particles saved.')
    return smass,smetarray,sage,gmass,gsfr,gmetarray,gtemp,gnh,gpos,bhmass,bhmdot,pZstar,pCstar,pOstar,pZgas,pCgas,pOgas

# class that holds the particle data for a galaxy
class Gal_PartData:
    def __init__(self, ID):
        self.id = ID
        self.smass = 0
        self.sage = 0
        self.gmass = 0
        self.gsfr = 0
        self.gtemp = 0
        self.gnh = 0
        self.grad = 0
        self.bhmass = 0
        self.bhar = 0
        self.pZstar = 0
        self.pCstar = 0
        self.pOstar = 0
        self.pZgas = 0
        self.pCgas = 0
        self.pOgas = 0
    def sfr_weighted_Z(self):
        # class function that provides the SFR-weighted total metalicites of gas particles
        Z_warm = np.array([self.pZgas[i]/self.gsfr[i] for i in range(0,len(self.pZgas)) if self.gsfr[i]>0])
        Z_warm_tot = np.sum(self.pZgas*self.gsfr)/(np.sum(self.gsfr)+1.e-10)
        return Z_warm,Z_warm_tot
    def sfr_weighted_CO(self):
        # class function that provides the SFR-weighted C/O fraction of gas particles
        Z_ratio = self.pCgas/self.pOgas
        CO_warm = np.array([(Z_ratio[i]*self.gsfr[i])/(np.sum(self.gsfr)+1.e-10) for i in range(0,len(Z_ratio))])
        CO_warm_tot = np.sum(Z_ratio*self.gsfr)/(np.sum(self.gsfr)+1.e-10)
        return CO_warm,CO_warm_tot
    def mass_weighted_age(self):
        # class function that provides the mass-weighted age of stellar particles
        age_mw = np.zeros(len(self.sage))
        mass_t = np.sum(self.smass)
        for i in range(0, len(age_mw)):
            age_mw[i] = self.sage[i]*self.smass[i]/mass_t
        return age_mw
    def mass_weighted_Z(self):
        # class function that provides mass-weighted total metalicites of stellar particles
        Z_mw = np.zeros(len(self.pZstar))
        mass_t = np.sum(self.smass)
        for i in range(0, len(Z_mw)):
            Z_mw[i] = self.pZstar[i]*self.smass[i]/mass_t
        return Z_mw

# combine caesar galaxy info to particle info and return a list with each galaxy data
def part_to_gal(gals,snapfile,sim,metals,H0):
    gals_data  = []
    print('Connecting each particle with its galaxy...')
    for gal in gals:
        new_gal_data = Gal_PartData(gal.GroupID)
        starparts = gal.slist  # list of star particle IDs in gal
        gasparts = gal.glist  # list of gas particle IDs in gal
        bhparts = gal.bhlist  # list of BH particle IDs in gal  (can be more than 1)
        # collect gas info
        new_gal_data.gsfr = readsnap(snapfile,'sfr','gas',units=1,suppress=1)[gasparts]  # gas particle SFR Mo/yr
        gmetarray = readsnap(snapfile,'Metallicity','gas',units=1,suppress=1)[gasparts]  # gas metal array
        #new_gal_data.grad  = np.array([np.linalg.norm(gpos[k]-gal.pos.value) for k in gasparts])
        new_gal_data.pZgas = np.asarray([i[0] for i in gmetarray])
        new_gal_data.pCgas = np.asarray([i[metals[0]] for i in gmetarray])
        new_gal_data.pOgas = np.asarray([i[metals[1]] for i in gmetarray])
        # collect star info
        #new_gal_data.smass = np.array([smass[k] for k in starparts])
        new_gal_data.sage = readsnap(snapfile,'age','star',units=1,suppress=1)[starparts]
        new_gal_data.sage = a_to_t(new_gal_data.sage,sim.simulation.omega_matter,H0)  # age in Gyr
        smetarray = readsnap(snapfile,'Metallicity','star',units=1,suppress=1)[gasparts]  # gas metal array
        new_gal_data.pZstar = np.asarray([i[0] for i in smetarray])
        new_gal_data.pCstar = np.asarray([i[metals[0]] for i in smetarray])
        new_gal_data.pOstar = np.asarray([i[metals[1]] for i in smetarray])
        # collect BH and central gas info
        #pbhm = np.array([bhmass[k] for k in bhparts])
        #pbhar = np.array([bhmdot[k] for k in bhparts])
        #if len(pbhm)>0: 
            #new_gal_data.bhar = pbhar[np.argmax(pbhm)]  # BHAR for most massive BH in galaxy
            #new_gal_data.bhmass = pbhm.max()
        gals_data.append(new_gal_data)
    print('Data for particles and galaxies saved')
    output = open('part_data_'+str(sys.argv[3])+'.pkl','wb')
    pickle.dump(gals_data, output)
    output.close()
    print('Pickle file saved!')
    return gals_data
         
# function that creats histogram plot of stellar ages for a given galaxy weighted with the stellar mass
def s_age_histogram(gal, nbins, mass_weight=False):
    print('Making stellar ages histogram weighted with star particle mass...')
    if mass_weight:
        ages = gal.mass_weighted_age()
    else:
        ages = gal.sages
    bin_count, bin_edges = np.histogram(ages, bins=nbins)#, weights=gal.smass)
    bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])
    fig = plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.plot(bin_cent, np.log10(bin_count/float(len(gal.sage))), 'k-')
    ax.set_xlabel('Age (Gyr)', fontsize=16)
    ax.set_ylabel(r'$\log(N/N_{star})$', fontsize=16)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    if mass_weight:
        fig.savefig('./stellar_plots/s_age_mw_histogram_'+str(sys.argv[3])+'.png', format='png', dpi=250, bbox_inches='tight')
    else:
        fig.savefig('./stellar_plots/s_age_histogram_'+str(sys.argv[3])+'.png', format='png', dpi=250, bbox_inches='tight')
    print('Stellar ages histogram done!')

# function that creates histogram plot of stellar metalicity 
def s_Z_histogram(gal, nbins, mass_weight=False):
    print('Making histogram for the total metalicities of stars...')
    if mass_weight:
        Zs = gal.mass_weighted_Z()
    else:
        Zs = gal.pZstar
    bin_count, bin_edges = np.histogram(Zs, bins=nbins)#, weights=gal.smass)
    bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])
    fig = plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.plot(bin_cent, np.log10(bin_count/float(len(gal.pZstar))), 'k-')
    ax.set_xlabel('Z', fontsize=16)
    ax.set_ylabel(r'$\log(N/N_{star})$', fontsize=16)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    if mass_weight:
        fig.savefig('./stellar_plots/s_z_mw_histogram_'+str(sys.argv[3])+'.png', format='png', dpi=250, bbox_inches='tight')
    else:
        fig.savefig('./stellar_plots/s_z_histogram_'+str(sys.argv[3])+'.png', format='png', dpi=250, bbox_inches='tight')
    print('Stellar metalicities histogram done!')

if __name__ == '__main__':
    
    snapfile,caesarfile,metal1,metal2 = obtain_snap_shell()
    s_gal = np.array([4973,8845,9553])
    metals = np.array([metal1,metal2])
    sim, redshift, h, cosmo, thubble, H0, rhocrit, mlim, gals = read_caesar(caesarfile,s_gal)
    print(len(gals))
    #smass,smetarray,sage,gmass,gsfr,gmetarray,gtemp,gnh,gpos,bhmass,bhmdot,pZstar,pCstar,pOstar,pZgas,pCgas,pOgas = read_snap(snapfile,sim,metals)
    
    gals_data = part_to_gal(gals,snapfile,sim,metals,H0)
    #obj = open('./part_data_'+str(sys.argv[3])+'.pkl','rb')
    #gals_data = pickle.load(obj)
    #s_age_histogram(gals_data[0],15,mass_weight=True)
    #s_Z_histogram(gals_data[0],15,mass_weight=False)


