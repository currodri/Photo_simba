import numpy as np
from astroML.resample import jackknife
A4_width = 6.0 #8 inch
mydpi    = 360


from matplotlib.colors import LinearSegmentedColormap
#Red, Green, Blue-------------------------
RGB_       = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
my_rgb     = LinearSegmentedColormap.from_list(
         'RGB', RGB_, N=256)
#Reversing the order of the colors
my_rgb_r   = LinearSegmentedColormap.from_list(
         'RGB', RGB_[::-1], N=256)
#------------------------------------------


def sche_func_HI(m_HI):
    #m_HI: M_sun
    #ALFALFA
    m_HI = np.log10(m_HI)
    phi_s  = -2.31875876262 #np.log10(0.0048)
    m_s    = 9.96
    alpha  = -1.29
    return 0.3622156 + phi_s + (alpha+1)*(m_HI-m_s) - ( 10 ** m_HI/10**m_s)




def sche_func_S(m_S):
    '''
    m_S: stellar mass M_sun
    '''
    #Baldry 2008
    ms = 10**10.648
    phi1 = 0.00426
    phi2 = 0.00058
    al1  = -0.46
    al2  = -1.58
    #Baldry 2012, GAMMA
    ms = 10**10.66
    phi1 = 0.00396
    phi2 = 0.00079
    al1  = -0.35
    al2  = -1.47

    ratio = m_S/ms
    return np.exp(-ratio) * (phi1*ratio**al1 + phi2*ratio**al2)


def quenched(sfr, mass, z):
    '''
    output = quenched( sfr, mass, z)
    1: quenched
    0: non-quenched
    Based on Noeske et al. 2007
    sfr: Msun/yr
    mass: stellar mass Msun
    z: redshift
    '''
    lmass = np.log10(mass)
    x     = np.array([np.min(lmass), np.max(lmass)])
    lsfr = np.log10(sfr.clip(1e-5))
    lim  = -1.29 + 0.65*(lmass-10) + 1.33*(z-0.1)
    y    = np.array([np.min(lim), np.max(lim)])
    return np.where(lsfr<lim, 1, 0), x, y
    

def completeness_lim(Mstar, z, q):
    '''
    return boolean array of the possibly detected galaxies
    based on PRIMUS, Berti-16 Table 1
    z: redshift
    q: 1 quenched, 0 star forming
    completeness_lim(mstar, z, q)
    '''
    l = len(Mstar)
    if z<0.2: return np.ones(l).astype(bool) #all are detected
    if z<0.8:
        sf_lim = 10**np.median([10.63, 10.12, 10.90, 9.89, 10.10])
        q_lim  = 10**np.median([10.71, 10.52, 10.79, 10.60, 10.43])
    if z<0.65:
        sf_lim = 10**np.median([10.44, 9.75, 10.59, 9.58, 9.77])
        q_lim  = 10**np.median([10.44, 10.22, 10.55, 10.22, 10.13])
    if z<0.5:
        sf_lim = 10**np.median([10.19, 9.38, 10.25, 9.30, 9.44])
        q_lim  = 10**np.median([10.17, 9.89, 10.30, 9.85, 9.85])
    if z<0.4:
        sf_lim = 10**np.median([9.92, 9.05, 9.94, 9.06, 9.13])
        q_lim  = 10**np.median([9.92, 9.58, 10.06, 9.52, 9.61])
    if z<0.3:
        sf_lim = 10**np.median([9.60, 8.68, 9.58, 8.80, 8.79])
        q_lim  = 10**np.median([9.65, 9.23, 9.80, 9.17, 9.35])
    if z>=0.8:
        sf_lim = 10**np.median([10.69, 10.46, 11.14, 10.21, 10.38])
        q_lim  = 10**np.median([10.96, 10.75, 10.99, 10.96, 10.73])
    output = np.zeros(l)
    output[q&(Mstar>q_lim)]     = 1 #if quenched, only above q_lim
    output[(~q)&(Mstar>sf_lim)] = 1 #if star forming, above sf_lim
    print 'Based on PRIMUS: sf M*_lim = %.2f'%np.log10(sf_lim)
    print '                 q  M*_lim = %.2f'%np.log10(q_lim)
    return output.astype(bool)




def distance(x0, x1, dimensions, p): 
    '''
    distance(x0, x1, dimensions, p)
    x0: point coordinates
    x1: array(1d, or 3d)
    p : 0 if 1d or else if higher d
    dimensions: boxsize
    '''
    delta = x1 - x0
    delta = np.where(delta >  0.5 * dimensions, delta - dimensions, delta)
    delta = np.where(delta < -0.5 * dimensions, delta + dimensions, delta)
    if p==0:
        return delta
    if p>0:
        return np.sqrt((delta ** 2).sum(axis=-1))

def DN(pos,box,depth0,depth1,flag):
    gl = np.shape(pos)
    dN = np.zeros(gl)
    dM = np.zeros(gl)
    
    rem = np.array(([1,2],[0,2],[0,1]), dtype=np.int) #x,y,z projection
    for i in range(gl[0]):
        for j in range(3):
            #remove galaxies outside the depth
            cut = distance(pos[i,j], (pos[:,j]), box, 0)
            newgal = (pos)[(cut<=depth1)&(cut>=depth0)] #condition
            distance_ = distance(pos[i,rem[j]], newgal[:,rem[j]], box, 1)
            #for the flag neighbor
            R         = distance_[np.argsort(distance_)[int(flag)]]
            dN[i,j]   = float(flag)/(np.pi*R**2.)#flag^th neighbor
            dM[i,j]   = len(distance_[distance_<1])-1 #distance <1Mpc
    return dN,dM




def mass_function(mass, vol_Mpc, nbin=30, minmass=6):
        #not very useful, used in cosmic_variance()
        maxmass     = np.log10(np.nanmax(mass))
        lbin        = np.linspace(minmass, maxmass, nbin+1)
        bin_        = 10** lbin
        step        = ( maxmass - minmass ) / nbin
        x           = 10 ** ( (lbin + step/2.)[:nbin])
        hist        = np.histogram( mass, bins=bin_, range=(bin_.min(),bin_.max()) )[0]
        y           =  hist / (vol_Mpc * step)
        return x, y, bin_, step


def cosmic_variance(mass, pos, boxsize, vol_Mpc, nbin=30, minmass=6):
        '''
        mass: M_sun
        '''
        pos       = np.floor(pos/(0.5*boxsize)).astype(np.int)
        gal_index = pos[:,0] + pos[:,1]*2 + pos[:,2]*4
        x, y, bin_, step = mass_function(mass, vol_Mpc, nbin=nbin, minmass=minmass)
        store_mf  = np.zeros((nbin, 8)) 
        for i0 in xrange(8):
            m_s_ = mass[gal_index==i0]
            if len(m_s_) < 1 : continue
            _x, _y, _bin, _step = mass_function(m_s_, vol_Mpc, nbin=nbin, minmass=minmass)
            phi_  = np.log10(8*_y)
            phi_  = np.where(phi_<-100, np.log10(y), phi_)
            store_mf[:,i0] = phi_
        store_mf  = np.ma.masked_invalid(store_mf)
        var  = np.ma.std(store_mf, axis=1)
        return x, y, var 


def conv_eff(mass, z): 
    '''
    Moster-2013
    Equation 2
    Equations 11,12,13,14. Table 1
    '''
    N    = 0.0351-0.0247*z/(z+1)
    M1   = 10**(11.590+1.195*z/(z+1))
    beta = 1.376-0.826*z/(z+1)
    gama = 0.608+0.329*z/(z+1)
    print "log(Mmax/Msun) = %.2f at z=%.2f"%(np.log10(M1*(beta/gama)**(1/(beta+gama))), z)
    return 2.0*N/((mass/M1)**(-beta)+(mass/M1)**gama)


def median_value(x,y, nbin=6, xmin= None, xmax = None, err=1 ):
    if xmin==None: xmin = x.min()
    if xmax==None: xmax = x.max()
    step = (xmax - xmin)/nbin
    bin_ = np.linspace(xmin-step/1.99, xmax+step/1.99, nbin+2)
    mid_bin  = (bin_+step/2.)[:nbin+1]
    dig  = np.digitize(x,bin_)
    if not err: return mid_bin, np.array([np.median(y[dig==i+1]) for i in xrange(len(mid_bin))])
    yout = np.zeros(len(mid_bin))
    yerr = np.zeros(len(mid_bin))
    std  = np.zeros(len(mid_bin))
    for i in xrange(len(mid_bin)):
        s = y[dig==i+1]
        s = s[(~np.isnan(s))&(np.isfinite(s))]
        if not len(s):continue
        mu, sig   = jackknife(s, np.mean, kwargs=dict(axis=1))
        yerr[i]   = sig
        yout[i]   = mu
        std[i]    = np.std(s)

    
    return mid_bin, yout, yerr, std #np.array([np.mean(y[dig==i+1]) for i in xrange(len(mid_bin))])

def mean_value(x,y, nbin=6, xmin= None, xmax = None):
    if xmin==None: xmin = x.min()
    if xmax==None: xmax = x.max()
    step = (xmax - xmin)/nbin
    bin_ = np.linspace(xmin-step/1.99, xmax+step/1.99, nbin+2)
    mid_bin  = (bin_+step/2.)[:nbin+1]
    dig  = np.digitize(x,bin_)
    
    return mid_bin, np.array([np.mean(y[dig==i+1]) for i in xrange(len(mid_bin))])

def create_ax(ratio, x_p=1, y_p=1, fonts=12):
    '''
    ax = create_ax(ratio, args)
    ratio: 1: cover entire page
           0.5: half page
           ...
    x_p: number of subplot on column
    y_p: number of subplot on raw

    x_p = 2, y_p = 2:
       ax2     ax3
       ax0     ax1


    '''
    import matplotlib.pyplot as mpl
    mpl.rcParams['xtick.major.size']  = 5
    mpl.rcParams['xtick.major.width'] = .5
    mpl.rcParams['ytick.major.size']  = 5
    mpl.rcParams['ytick.major.width'] = .5
    # ...
    mpl.rcParams['xtick.minor.size']  = 3
    mpl.rcParams['xtick.minor.width'] = .5
    mpl.rcParams['ytick.minor.size']  = 3
    mpl.rcParams['ytick.minor.width'] = .5
    
    
    #mpl.rcParams['ps.useafm'] = True
    #mpl.rcParams['pdf.use14corefonts'] = True
    #mpl.rcParams['text.usetex'] = True
    
    font   = {'family' : 'monospace',
              'weight' : 'normal',#can be bold
              'size'   : fonts}
    mpl.rc('font', **font)
    #mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':fonts})


    margin_x       = 0.1
    margin_y       = 0.1
    width_ax       = (1.0-margin_x)/x_p
    height_ax      = (1.0-margin_y)/y_p

    plot_width     = ratio*A4_width
    plot_height    = plot_width/x_p*y_p
    fig    = mpl.figure(figsize=(plot_width, plot_height), dpi=mydpi)

    if x_p==1 and y_p==1:
        output = fig.add_axes([margin_x,margin_y, width_ax, height_ax])
        output.tick_params(axis='both', which='major', labelsize=fonts-4)
        return output


    output = []
    for i in xrange(y_p):
        for j in xrange(x_p):
            sub_ax = fig.add_axes([margin_x+j*width_ax, margin_y+i*height_ax, width_ax, height_ax])
            sub_ax.tick_params(axis='both', which='major', labelsize=fonts-4)
            output.append(sub_ax)

    return output
