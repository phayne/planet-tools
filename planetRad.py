"""
Radiation and radiative transfer tools for planetary science
"""

# Physical constants
sigma = 5.670367e-8 # Stefan-Boltzmann constant [W.m-2.K-4]

# Constants for wavelength-based Planck function
c1 = 1.191043e-16 # 2*h*c**2
c2 = 0.01438777   # h*c/k

# Constants for frequency-based Planck function
a1 = 1.191066e-5
a2 = 1.438833

# NumPy is needed for various math operations
import numpy as np
np.seterr(over='ignore') # suppress overflow warnings

# Valid temperature range and increment
TMIN =  1.0 # Minimum valid temperature [K]
TMAX = 600.0 # Maximum valid temperature [K]
DT = 1.0 # Temperature increment/precision [K]
TRANGE = np.arange(TMIN, TMAX, DT) # Temperature array

##########
# planck #
##########
# Planck's function for blackbody radiance in SI-wavelength units:
# Input:
#    T = temperature [K]
#    lam = wavelength [m]
# Output:
#    spectral radiance [W.m-2.sr-1.m-1]
def planck(T, lam):
    r = c1*lam**-5 / (np.exp(c2*(lam*T)**-1) - 1)
    return r

############
# planckwn #
############
# Planck's function for blackbody radiance in frequency units:
# Input:
#    T = temperature [K]
#    nu = frequency [cm-1]
# Output:
#    spectral radiance [mW.m-2.sr-1.cm]
def planckwn(T, nu):
    r = a1*nu**3 / (np.exp(a2*nu/T) - 1)
    return r

###########
# iplanck #
###########
# Inverse of Planck's function in SI-units:
# Input:
#    R = radiance [W.m-2.sr-1.m-1]
#    lam = wavelength [m]
# Output:
#    brightness temperature [K]
def iplanck(R, lam):
    Tb = c2 / ( lam * np.log( (c1/(lam**5*R)) + 1 ) )
    return Tb

########################
# effective_wavelength #
########################
# Radiance- and response-weighted wavelength for a
# given spectral response function and scene temperature
# Input:
#    f = spectral response function
#    lam = wavelength in METERS
#    T = scene temperature in Kelvin
# Output:
#    Effective wavelength in meters
def effective_wavelength(f, lam, T):
    b = planck(T,lam)
    lam_eff = np.trapz(lam*f*b,lam)/np.trapz(f*b,lam)
    
    return lam_eff
########################

##################
# radiance_to_tb #
##################
# Convert a measured radiance to brightness
# temperature (equivalent blackbody temperature), 
# for the given spectral response function
# Input:
#    f = spectral response function
#    lam = wavelength in METERS
#    radiance = radiance in SI units: W.m-2.sr-1.m-1
# Output:
#    brightness temperature [K]
def radiance_to_tb(f, lam, radiance):
    b = tb_to_radiance(f, lam, TRANGE)
    tb = np.interp(radiance, b, TRANGE)
    
    return tb

##################
# tb_to_radiance #
##################
# Convert a brightness temperature (equivalent 
# blackbody temperature) to radiance, for the 
# given spectral response function
# Input:
#    f = spectral response function
#    lam = wavelength in METERS
#    tb = brightness temperature [K]
# Output:
#    radiance [W.m-2.sr-1.m-1]
def tb_to_radiance(f, lam, tb):
    # Check length of tb
    if hasattr(tb, "__len__"):
        radiance = []
        for T in tb:
            b = planck(T, lam)
            radiance.append( np.trapz(f*b,lam) )
    else:
        b = planck(tb, lam)
        radiance = np.trapz(f*b,lam)
     
    return radiance

###########################
# equilibrium_temperature #
###########################
# Equilibrium temperature for rapid rotator
# Assumes constant surface temperature, i.e. effectively
# infinite thermal inertia over the thermal skin depth
# Input:
#    F = incident solar flux [W.m-2]
#    A = bolometric solar albedo (i.e., Bond albedo)
#    e = thermal bolometric emissivity
# Output:
#    T = temperature [K]
def equilibrium_temperature(F, A, e):
    return ((1-A)*F/(4*e*sigma))**0.25

##########################
# albedo_layer_twostream #
##########################
# Calculate albedo** for a layer of finite optical
# thickness and known optical properties, overlying
# a layer of known albedo. This model uses the
# well-established delta-Eddington two-stream method to
# approximate radiative transfer in the two-layer medium.
#
# **The quantity calculated is the "directional-hemispheric
# reflectance" or "hemispheric albedo" (see: Hapke, 1981,
# J. Geophys. Res., v. 86, p. 3049). The approach taken
# here is that of Wiscombe and Warren (1980), J. Atmos.
# Sci., v. 37, pp. 2712-2733. The layer is assumed to
# be composed of discrete particles (grains), whose
# optical properties can be approximated using Mie
# theory.
# 
# Input:
#    tau = optical thickness of the layer [scalar or array]
#    w0 = single-scattering albedo of grains [scalar]
#    g = asymmetry parameter, g = <cos(phase)> [scalar]
#    mu = cosine of the emission angle [scalar]
#    asurf = albedo of underlying surface
# Output:
#    directional-hemispheric reflectance
@np.vectorize
def albedo_layer_twostream(tau, w0, g, mu, asurf):
    
    # Cutoff criterion for semi-infinite approx.
    CUTOFF_H = 1e3
    CUTOFF_L = -1e3
    
    # DEBUG
    #print(w0p/(1+p) * (1-bp*x*mu)/(1+x*mu))
    
    # Transform to delta-Eddington coordinates
    g2 = g**2
    taup = (1-w0*g2)*tau
    w0p = (1-g2)*w0/(1-g2*w0)
    gp = g/(1+g)
    
    # Define some useful parameters
    ap = 1 - w0p*gp
    bp = gp/ap
    x = np.sqrt(3*ap*(1-w0p))
    p = (2/3)*x/ap
    gamma = (1-asurf)/(1+asurf)
    qp = (gamma+p)*np.exp(+x*taup)
    qm = (gamma-p)*np.exp(-x*taup)
    q = (1+p)*qp - (1-p)*qm
    
    # Check whether the layer is optically thick;
    # if so, use a simplified formula
    if ( qp>CUTOFF_H or qm<CUTOFF_L ) :
        albedo = w0p/(1+p) * (1-bp*x*mu)/(1+x*mu)
    # Otherwise, use the full formula
    else:
        albedo = (1/q) * ( 2*( p*(1-gamma + w0p*bp) + w0p*(1+bp)*(gamma*x*mu - p)/(1 - (x*mu)**2))  \
                             * np.exp(-taup/mu) - w0p*bp*(qp-qm) \
                             + w0p*(1+bp)*(qp/(1+x*mu) - qm/(1-x*mu)) ) 
    
    return albedo