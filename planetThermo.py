"""
Thermodyanmic functions and constants for planetary science
"""

# Load Python dependencies
import numpy as np

######################
# Physical constants #
######################

# Pi
pi = np.pi

# Proton mass
mp = 1.6726e-27

# Molecular masses [kg]
m_h2o = 3.0132e-26

# Molar masses [kg.mole-1]
mw_h2o = 18.015e-3   # Water

# Thermodynamic constants
R = 8.31447 # Universal gas constant [J.mol-1.K-1]
k_B = 1.38064852e-23 # Boltzmann constant [m2.kg.s-2.K-1]

class volatile:
    '''
    A volatile is a molecular species that undergoes
    phase transitions at planetary surface and/or atmospheric
    temperatures
    '''
    #__repr__ object prints out a help string when help is
    #invoked on the volatile object or the volatile name is typed
    def __repr__(self):
        line1 =\
        'This volatile object contains information on %s\n'%self.name
        line2 = 'Type \"help(volatile)\" for more information\n'
        return line1+line2
    def __init__(self):
        self.name = None #Name of the volatile

########
# ph2o #
########
# -----------------------------------------------
# Equilibrium vapor pressure over water over ice
# Marti and Mauersberger (1993)
# -----------------------------------------------
# Input:
#    T = temperature of ice [K]
# Output:
#    vapor pressure [Pa]
def ph2o(T):
    A = -2663.5
    B = 12.537
    p = 10**(A/T + B)

    return p

########
# pco2 #
########
# --------------------------------------------
# Equilibrium vapor pressure over solid CO2
# Brown and Ziegler (1980)
# --------------------------------------------
# Input:
#    T = temperature of solid [K]
# Output:
#    vapor pressure [Pa]
def pco2(T):
    A0 = 2.13807649*10**1
    A1 = -2.57064700*10**3
    A2 = -7.78129489*10**4
    A3 = 4.32506256*10**6
    A4 = -1.20671368*10**8
    A5 = 1.34966306*10**9

    # Pressure in torr
    ptorr = np.exp(A0 + A1/T + A2/T**2 + A3/T**3 + A4/T**4 + A5/T**5)

    # Pressure in Pa
    p = 133.3223684211*ptorr

    return p

########
# th2o #
########
# --------------------------------------------
# Equilibrium H2O frost point at given partial
# pressure
# Marti and Mauersberger (1993)
# --------------------------------------------
# Input:
#    p = vapor pressure [Pa]
# Output:
#    temperature of solid [K]
def th2o(p):
    # Definitions
    dT = 0.1 # Temperature precision
    Tmin = 20 # Minimum temperature to try
    Tmax = 600.0 # Maximum temperature to try
    
    # Temperature and pressure arrays
    T_arr = np.arange(Tmin, Tmax, dT)
    p_arr = ph2o(T_arr)
    
    # Interpolate to find T
    T = np.interp(p, p_arr, T_arr)
    
    return T

########
# tco2 #
########
# --------------------------------------------
# Equilibrium CO2 frost point at given partial
# pressure
# Brown and Ziegler (1980)
# --------------------------------------------
# Input:
#    p = vapor pressure [Pa]
# Output:
#    temperature of solid [K]
def tco2(p):
    # Definitions
    dT = 0.1 # Temperature precision
    Tmin = 20 # Minimum temperature to try
    Tmax = 200.0 # Maximum temperature to try
    
    # Temperature and pressure arrays
    T_arr = np.arange(Tmin, Tmax, dT)
    p_arr = pco2(T_arr)
    
    # Interpolate to find T
    T = np.interp(p, p_arr, T_arr)
    
    return T

######################
# h2oSublimationRate #
######################
# -------------------------------------------------------------------------
# Water ice sublimation rate, in kg.m-2.s-1, based on planar surface
# solution from Estermann (1955): mdot = pv*sqrt(mw_h2o/2piRT), where mh2o
# is the molecular weight of water, R is the universal gas constant, T is
# temperature in Kelvin, and pv is the saturation vapor pressure. This is
# the upper limit on sublimation rate, where the actual sublimation rate
# depends on the difference (p-pv) and the rate of adsorption.
# -------------------------------------------------------------------------
# Input:
#    T = temperature of ice [K]
# Output:
#    sublimation rate [kg.m-2.s-1]
def h2oSublimationRate(T):
    # Equilibrium water vapor pressure over ice
    pv = ph2o(T) # [Pa]

    # Sublimation rate
    mdot = pv * np.sqrt( mw_h2o/(2*pi*R*T) ) # [kg.m-2.s-1]

    return mdot
