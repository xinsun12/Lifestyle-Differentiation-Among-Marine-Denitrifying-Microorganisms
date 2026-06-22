# -*- coding: utf-8 -*-
"""
Updated April 2026

Purpose
-------
    A 0D chemostat model of redox reactions occuring in marine OMZs with a focus on N2O and denitrification!
    Modular denitrification included, 
    yields of denitrifiers depend on Gibbs free energy,
    copiotrophs and oligotrophs are included
"""

#%% imports
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten

# numerical packages
from numba import jit

#%% Set initial conditions and incoming concentrations to chemostat experiment

### Organic matter (S)
## range of org C flux from 2-8 uM C m-2 day-1 --> 0.26 - 1 uM N m-3 day-1, 
## assuming 1 m3 box and flux into top of box
Sd0_exp = 0.05 #default

# OM pulse conditions
xpulse_Sd = np.arange(0.1,22.1,0.5) #Default

xpulse_int = 50# Default

### Oxygen supply
O20_exp = np.array([0])

### other default parameters
in_an = 0
N2Oammonia = 0 
K_n_Den4 = 4 
K_n_Den2 = K_n_Den4 
GrowthRateMM = 1 
Gasuptake = 3 

### model parameters for running experiments (dil = 0.04 d-1)
dil = 0.04  # dilution rate (1/day); 
days = 5e4
dt = 0.001  # timesteps length (days)
timesteps = days/dt     # number of timesteps
out_at_day = dt# output results this often
nn_output = days/out_at_day     # number of entries for output
nn_outputforaverage = int(2000/out_at_day) # finish value is the average of the last XX (number) of outputs


#%% # 
outputd1 = xpulse_Sd 
outputd2 = O20_exp

#%% initialize arrays for output

# Nutrients
fin_O2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_Sd = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_Sp = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO3 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NH4 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2O = np.ones((len(outputd1), len(outputd2))) * np.nan 

# Biomasses
fin_bHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bFac = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
# add copiotrophic (1 aero + 6 denitrifiers)
fin_bHetC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b1DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b2DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b3DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b4DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b5DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b6DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
# end
fin_bAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bNOOan = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bAOX = np.ones((len(outputd1), len(outputd2))) * np.nan

# Growth rates
fin_uHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uFac = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
# add Copio 1 aero + 6 copiotrophic denitrifiers
fin_uHetC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u1DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u2DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u3DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u4DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u5DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u6DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
# end
fin_uAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uNOOan = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uAOX = np.ones((len(outputd1), len(outputd2))) * np.nan

# track facultative average respiration
fin_facaer = np.ones((len(outputd1), len(outputd2))) * np.nan

# Rates 
fin_rHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rHetAer = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rO2C = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
#fin_r7Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
# end
fin_rAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rNOOan = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rAOX = np.ones((len(outputd1), len(outputd2))) * np.nan

fin_rN2Oammonia = np.ones((len(outputd1), len(outputd2))) * np.nan

#%% set traits of the different biomasses
from traits_OligoCopi_Clean import * 

#%% calculate R*-stars for all microbes
from O2_star_Xin import O2_star
from N2O_star_Xin import N2O_star 
from R_star_Xin import R_star
from R_star_Xin import R_starMM2



# O2 (nM-O2) 
O2_star_aer = R_star(dil, K_o2_aer, mumax_Het / y_oO2, y_oO2) * 1e3 
O2_star_fac = R_star(dil, K_o2_aer, mumax_Het * fac_penalty / y_oO2Fac, y_oO2Fac) * 1e3 
O2_star_aoo = R_star(dil, K_o2_aoo, mumax_AOO / y_oAOO, y_oAOO) * 1e3 
O2_star_noo = R_star(dil, K_o2_noo, mumax_NOO / y_oNOO, y_oNOO) * 1e3 
# N2O (nM-N)
N2O_star_den5 = R_star(dil, K_n2o_Den, VmaxN_5Den, y_n5N2O) * 1e3
# OM
OM_star_aer = R_star(dil, K_s, VmaxS, y_oHet)
OM_star_aerC = R_star(dil, K_sC, VmaxSC, y_oHet)
OM_star_den1 = R_star(dil, K_s, VmaxS, y_n1Den)
OM_star_den2 = R_star(dil, K_s, VmaxS, y_n2Den)
OM_star_den3 = R_star(dil, K_s, VmaxS, y_n3Den)
OM_star_den4 = R_star(dil, K_s, VmaxS, y_n4Den)
OM_star_den5 = R_star(dil, K_s, VmaxS, y_n5Den)
OM_star_den6 = R_star(dil, K_s, VmaxS, y_n6Den)
#add 6 copiotrophic denitrifiers
OM_star_den1C = R_star(dil, K_sC, VmaxSC, y_n1Den)
OM_star_den2C = R_star(dil, K_sC, VmaxSC, y_n2Den)
OM_star_den3C = R_star(dil, K_sC, VmaxSC, y_n3Den)
OM_star_den4C = R_star(dil, K_sC, VmaxSC, y_n4Den)
OM_star_den5C = R_star(dil, K_sC, VmaxSC, y_n5Den)
OM_star_den6C = R_star(dil, K_sC, VmaxSC, y_n6Den)
# Ammonia
Amm_star_aoo = R_star(dil, K_n_AOO, VmaxN_AOO, y_nAOO)
Amm_star_aox = R_star(dil, K_nh4_AOX, VmaxNH4_AOX, y_nh4AOX)
# Nitrite
nitrite_star_den2 = R_star(dil, K_n_Den2, VmaxN_2Den, y_n2NO2)
nitrite_star_den4 = R_star(dil, K_n_Den4, VmaxN_4Den, y_n4NO2)
nitrite_star_noo = R_star(dil, K_n_NOO, VmaxN_NOO, y_nNOO)
nitrite_star_nooan = R_star(dil, K_n_NOO, VmaxN_NOO, y_nNOOan)
nitrite_star_aox = R_star(dil, K_no2_AOX, VmaxNO2_AOX, y_no2AOX)
# Nitrate
nitrate_star_fac = R_star(dil, K_n_Den, VmaxN_1DenFac, y_n1NO3Fac)
nitrate_star_den1 = R_star(dil, K_n_Den, VmaxN_1Den, y_n1NO3)
nitrate_star_den3 = R_star(dil, K_n_Den, VmaxN_3Den, y_n3NO3)
nitrate_star_den6 = R_star(dil, K_n_Den, VmaxN_6Den, y_n6NO3)
# copiotrophic den for N
nitrite_star_den2C = R_star(dil, K_n_DenC, VmaxN_DenC, y_n2NO2)
nitrite_star_den4C = R_star(dil, K_n_DenC, VmaxN_DenC, y_n4NO2)
nitrate_star_den1C = R_star(dil, K_n_DenC, VmaxN_DenC, y_n1NO3)
nitrate_star_den3C = R_star(dil, K_n_DenC, VmaxN_DenC, y_n3NO3)
nitrate_star_den6C = R_star(dil, K_n_DenC, VmaxN_DenC, y_n6NO3)
N2O_star_den5C = R_star(dil, K_n2o_DenC, VmaxN_DenC, y_n5N2O) * 1e3


#%% round up function
import math
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


#%%load the variables
print("choose the right folder")
#%%load the variables
# Set the directory where your files are stored
os.chdir("output=dt-Pulse22Oxy10withaerNOO-5e4_withCopioCN_Oxy0_highresolution")

# File name prefix
fname = 'OMpulse_P0.16'

xpulse_Sd = np.loadtxt(fname+'_pulse.txt', delimiter='\t')
O20_exp = np.loadtxt(fname+'_O2supply.txt', delimiter='\t')
fin_O2 = np.loadtxt(fname+'_O2.txt', delimiter='\t')
fin_N2 = np.loadtxt(fname+'_N2.txt', delimiter='\t')
fin_N2O = np.loadtxt(fname+'_N2O.txt', delimiter='\t')
fin_NO3 = np.loadtxt(fname+'_NO3.txt', delimiter='\t')
fin_NO2 = np.loadtxt(fname+'_NO2.txt', delimiter='\t')
fin_NH4 = np.loadtxt(fname+'_NH4.txt', delimiter='\t')
fin_Sd = np.loadtxt(fname+'_OM.txt', delimiter='\t')

fin_bHet = np.loadtxt(fname+'_bHet.txt', delimiter='\t')
fin_b1Den = np.loadtxt(fname+'_b1Den.txt', delimiter='\t')
fin_b2Den = np.loadtxt(fname+'_b2Den.txt', delimiter='\t')
fin_b3Den = np.loadtxt(fname+'_b3Den.txt', delimiter='\t')
fin_b4Den = np.loadtxt(fname+'_b4Den.txt', delimiter='\t')
fin_b5Den = np.loadtxt(fname+'_b5Den.txt',  delimiter='\t')
fin_b6Den = np.loadtxt(fname+'_b6Den.txt', delimiter='\t')
fin_bAOO = np.loadtxt(fname+'_bAOO.txt', delimiter='\t')
fin_bNOO = np.loadtxt(fname+'_bNOO.txt', delimiter='\t')
fin_bAOX = np.loadtxt(fname+'_bAOX.txt', delimiter='\t')

fin_uHet = np.loadtxt(fname+'_uHet.txt', delimiter='\t')
fin_u1Den = np.loadtxt(fname+'_u1Den.txt', delimiter='\t')
fin_u2Den = np.loadtxt(fname+'_u2Den.txt', delimiter='\t')
fin_u3Den = np.loadtxt(fname+'_u3Den.txt', delimiter='\t')
fin_u4Den = np.loadtxt(fname+'_u4Den.txt', delimiter='\t')
fin_u5Den = np.loadtxt(fname+'_u5Den.txt', delimiter='\t')
fin_u6Den = np.loadtxt(fname+'_u6Den.txt', delimiter='\t')
fin_uAOO = np.loadtxt(fname+'_uAOO.txt', delimiter='\t')
fin_uNOO = np.loadtxt(fname+'_uNOO.txt', delimiter='\t')
fin_uAOX = np.loadtxt(fname+'_uAOX.txt', delimiter='\t')

fin_rHet = np.loadtxt(fname+'_rHet.txt', delimiter='\t')
fin_rHetAer = np.loadtxt(fname+'_rHetAer.txt', delimiter='\t')
fin_r1Den = np.loadtxt(fname+'_r1Den.txt', delimiter='\t')
fin_r2Den = np.loadtxt(fname+'_r2Den.txt', delimiter='\t')
fin_r3Den = np.loadtxt(fname+'_r3Den.txt', delimiter='\t')
fin_r4Den = np.loadtxt(fname+'_r4Den.txt', delimiter='\t')
fin_r5Den = np.loadtxt(fname+'_r5Den.txt', delimiter='\t')
fin_r6Den = np.loadtxt(fname+'_r6Den.txt', delimiter='\t')
fin_rAOO = np.loadtxt(fname+'_rAOO.txt', delimiter='\t')
fin_rNOO = np.loadtxt(fname+'_rNOO.txt', delimiter='\t')
fin_rAOX = np.loadtxt(fname+'_rAOX.txt', delimiter='\t')
fin_rO2C = np.loadtxt(fname+'_rO2C.txt', delimiter='\t')

fin_rN2Oammonia = np.loadtxt(fname+'_rN2Oammonia.txt', delimiter='\t')


fin_bHetC = np.loadtxt(fname+'_bHetC.txt', delimiter='\t')
fin_b1DenC = np.loadtxt(fname+'_b1DenC.txt', delimiter='\t')
fin_b2DenC = np.loadtxt(fname+'_b2DenC.txt', delimiter='\t')
fin_b3DenC = np.loadtxt(fname+'_b3DenC.txt', delimiter='\t')
fin_b4DenC = np.loadtxt(fname+'_b4DenC.txt', delimiter='\t')
fin_b5DenC = np.loadtxt(fname+'_b5DenC.txt',  delimiter='\t')
fin_b6DenC = np.loadtxt(fname+'_b6DenC.txt', delimiter='\t')

fin_u1DenC = np.loadtxt(fname+'_u1DenC.txt', delimiter='\t')
fin_u2DenC = np.loadtxt(fname+'_u2DenC.txt', delimiter='\t')
fin_u3DenC = np.loadtxt(fname+'_u3DenC.txt', delimiter='\t')
fin_u4DenC = np.loadtxt(fname+'_u4DenC.txt', delimiter='\t')
fin_u5DenC = np.loadtxt(fname+'_u5DenC.txt', delimiter='\t')
fin_u6DenC = np.loadtxt(fname+'_u6DenC.txt', delimiter='\t')


#%% begin loop of experiments !! change in_Sd = Sd0_exp[k] OR Sd0_exp

from model_Xin_GibbsEnergyAvgConc_OligoCopi import OMZredox

for k in np.arange(len(outputd1)):
    for m in np.arange(len(O20_exp)):
        print(k,m)
        
        # 1) Chemostat influxes (µM-N or µM O2)
        in_Sd = Sd0_exp #Sd0_exp[k]
        in_O2 = O20_exp[m]
        in_Sp = 0.0  
        in_NO3 = 30.0
        in_NO2 = 0.0
        in_NH4 = 0.0
        in_N2 = 0.0
        in_N2O = 0.0
        # initial conditions
        initialOM = in_Sd 
        initialNO2 = 0
        
        # 2) Initial biomasses (set to 0.0 to exclude, set 0.1 to turn on)
        in_bFac = 0 # turn off 
        
        in_bHet = 0.1
        in_b1Den = 0.1 # NO3-->NO2, cross-feed
        in_b4Den = 0.1 # NO2-->N2O, cross-feed
        in_b5Den = 0.1 # N2O-->N2, cross-feed
        in_b2Den = 0.1 # NO2-->N2
        in_b3Den = 0.1 # complete denitrifier
        in_b6Den = 0.1 # NO3-->N2O
        
        in_b7Den = 0# 
        # all copiotrophic heterotrophs
        in_copio = 0.1
        in_bHetC = in_copio
        in_b1DenC = in_copio # NO3-->NO2, cross-feed
        in_b4DenC = in_copio # N2O producer (NO2-->N2O), cross-feed
        in_b5DenC = in_copio # N2O consumer (N2O-->N2), cross-feed
        in_b2DenC = in_copio # NO2-->N2
        in_b3DenC = in_copio # complete denitrifier
        in_b6DenC = in_copio # N2O producer (NO3-->N2O)
        
        in_bAOO = 0.1
        in_bNOO = 0.1
        in_bNOOan = 0#
        in_bAOX = 0.1
        
        # pulse conditions        
        pulse_int = xpulse_int 
        pulse_Sd = xpulse_Sd[k]
        pulse_bHet = 0.00
        pulse_bFac = 0.00
        pulse_O2 = 0.0
       
      
        # Call main model
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
                           pulse_Sd, pulse_bHet, pulse_bFac, pulse_O2, pulse_int, \
                           po_aer, po_aoo, po_noo, \
                           K_o2_aer, K_o2_aoo, K_o2_noo, \
                           pn2o_den, K_n2o_Den, \
                           mumax_Het, fac_penalty, den_penalty, mumax_AOO, mumax_NOO, mumax_AOX, GrowthRateMM, Gasuptake, \
                           VmaxS, K_s, VmaxSC, K_sC, \
                           VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_4Den, VmaxN_5Den, VmaxN_6Den, VmaxN_1DenFac, K_n_Den, K_n_Den2, K_n_Den4,\
                           VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
                           VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
                           y_oHet, y_oO2, y_oHetFac, y_oO2Fac, \
                           y_n1DenFac, y_n1NO3Fac, y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_n4Den, y_n4NO2, y_n5Den, y_n5N2O, y_n6Den, y_n6NO3, y_n7Den_NO3, y_n7NO3, e_n7Den_NO3, y_n7Den_N2O, y_n7N2O, e_n7Den_N2O,\
                           y_nAOO, y_oAOO, y_nNOO, y_nNOOan, y_oNOO, y_nh4AOX, y_no2AOX, \
                           e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, e_n4Den, e_n5Den, e_n6Den, e_n1Den, \
                           initialOM, initialNO2, in_Sd, in_Sp, in_O2, in_NO3, in_NO2, in_NH4, in_N2, in_N2O, in_an, \
                           in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bNOOan, in_bAOX, in_b4Den, in_b5Den, in_b6Den, in_b7Den,\
                           in_bHetC, in_b1DenC, in_b2DenC, in_b3DenC, in_b4DenC, in_b5DenC, in_b6DenC, \
                           N2Oammonia)
        
        out_Sd = results[0]
        out_Sp = results[1]
        out_O2 = results[2]
        out_NO3 = results[3]
        out_NO2 = results[4]
        out_NH4 = results[5]
        out_N2O = results[6] 
        out_N2 = results[7]
        out_bHet = results[8]
        out_bFac = results[9]
        out_b1Den = results[10]
        out_b2Den = results[11]
        out_b3Den = results[12]
        out_b4Den = results[13]
        out_b5Den = results[14]
        out_b6Den = results[15]
        #add copi (1 aerHet + 6 dens)
        out_bHetC = results[16]
        out_b1DenC = results[17]
        out_b2DenC = results[18]
        out_b3DenC = results[19]
        out_b4DenC = results[20]
        out_b5DenC = results[21]
        out_b6DenC = results[22]
        #end
        out_bAOO = results[23]
        out_bNOO = results[24]
        out_bAOX = results[25]
        out_uHet = results[26]
        out_uFac = results[27]
        out_u1Den = results[28]
        out_u2Den = results[29]
        out_u3Den = results[30]
        out_u4Den = results[31]
        out_u5Den = results[32]
        out_u6Den = results[33]
        #add copi (1 aerHet + 6 dens)
        out_uHetC = results[34]
        out_u1DenC = results[35]
        out_u2DenC = results[36]
        out_u3DenC = results[37]
        out_u4DenC = results[38]
        out_u5DenC = results[39]
        out_u6DenC = results[40]      
        #end
        out_uAOO = results[41]
        out_uNOO = results[42]
        out_uAOX = results[43]
        out_facaer = results[44]
        out_rHet = results[45]
        out_rHetAer = results[46]
        out_rO2C = results[47]
        out_r1Den = results[48]
        out_r2Den = results[49]
        out_r3Den = results[50]
        out_r4Den = results[51]
        out_r5Den = results[52]
        out_r6Den = results[53]
        out_rAOO = results[54]
        out_rNOO = results[55]
        out_rAOX = results[56]
        out_rN2Oammonia = results[57]
        
        out_bNOOan = results[58]
        out_uNOOan = results[59]
        out_rNOOan = results[60]
        
        out_b7Den = results[61]
        out_u7Den = results[62]
        #out_r7Den = results[63]
        
    
        # Record solutions in initialised arrays
        fin_O2[k,m] = np.nanmean(out_O2[-nn_outputforaverage::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-nn_outputforaverage::])
        fin_Sp[k,m] = np.nanmean(out_Sp[-nn_outputforaverage::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-nn_outputforaverage::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-nn_outputforaverage::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-nn_outputforaverage::])
        fin_N2[k,m] = np.nanmean(out_N2[-nn_outputforaverage::])
        fin_N2O[k,m] = np.nanmean(out_N2O[-nn_outputforaverage::]) 
        fin_bHet[k,m] = np.nanmean(out_bHet[-nn_outputforaverage::])
        fin_bFac[k,m] = np.nanmean(out_bFac[-nn_outputforaverage::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-nn_outputforaverage::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-nn_outputforaverage::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-nn_outputforaverage::])
        fin_b4Den[k,m] = np.nanmean(out_b4Den[-nn_outputforaverage::]) 
        fin_b5Den[k,m] = np.nanmean(out_b5Den[-nn_outputforaverage::]) 
        fin_b6Den[k,m] = np.nanmean(out_b6Den[-nn_outputforaverage::])
        fin_b7Den[k,m] = np.nanmean(out_b7Den[-nn_outputforaverage::]) 
        #add copi (1 aerHet + 6 dens)
        fin_bHetC[k,m] = np.nanmean(out_bHetC[-nn_outputforaverage::])
        fin_b1DenC[k,m] = np.nanmean(out_b1DenC[-nn_outputforaverage::])
        fin_b2DenC[k,m] = np.nanmean(out_b2DenC[-nn_outputforaverage::])
        fin_b3DenC[k,m] = np.nanmean(out_b3DenC[-nn_outputforaverage::])
        fin_b4DenC[k,m] = np.nanmean(out_b4DenC[-nn_outputforaverage::]) 
        fin_b5DenC[k,m] = np.nanmean(out_b5DenC[-nn_outputforaverage::]) 
        fin_b6DenC[k,m] = np.nanmean(out_b6DenC[-nn_outputforaverage::]) 
        #end
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-nn_outputforaverage::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-nn_outputforaverage::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-nn_outputforaverage::])
        fin_uHet[k,m] = np.nanmean(out_uHet[-nn_outputforaverage::])
        fin_uFac[k,m] = np.nanmean(out_uFac[-nn_outputforaverage::])
        fin_u1Den[k,m] = np.nanmean(out_u1Den[-nn_outputforaverage::])
        fin_u2Den[k,m] = np.nanmean(out_u2Den[-nn_outputforaverage::])
        fin_u3Den[k,m] = np.nanmean(out_u3Den[-nn_outputforaverage::])
        fin_u4Den[k,m] = np.nanmean(out_u4Den[-nn_outputforaverage::]) 
        fin_u5Den[k,m] = np.nanmean(out_u5Den[-nn_outputforaverage::]) 
        fin_u6Den[k,m] = np.nanmean(out_u6Den[-nn_outputforaverage::]) 
        fin_u7Den[k,m] = np.nanmean(out_u7Den[-nn_outputforaverage::])
        #add copi (1 aerHet + 6 dens)
        fin_uHetC[k,m] = np.nanmean(out_uHetC[-nn_outputforaverage::])
        fin_u1DenC[k,m] = np.nanmean(out_u1DenC[-nn_outputforaverage::])
        fin_u2DenC[k,m] = np.nanmean(out_u2DenC[-nn_outputforaverage::])
        fin_u3DenC[k,m] = np.nanmean(out_u3DenC[-nn_outputforaverage::])
        fin_u4DenC[k,m] = np.nanmean(out_u4DenC[-nn_outputforaverage::]) 
        fin_u5DenC[k,m] = np.nanmean(out_u5DenC[-nn_outputforaverage::]) 
        fin_u6DenC[k,m] = np.nanmean(out_u6DenC[-nn_outputforaverage::]) 
        #end
        fin_uAOO[k,m] = np.nanmean(out_uAOO[-nn_outputforaverage::])
        fin_uNOO[k,m] = np.nanmean(out_uNOO[-nn_outputforaverage::])
        fin_uAOX[k,m] = np.nanmean(out_uAOX[-nn_outputforaverage::])
        fin_facaer[k,m] = np.nanmean(out_facaer[-nn_outputforaverage::])
        fin_rHet[k,m] = np.nanmean(out_rHet[-nn_outputforaverage::])
        fin_rHetAer[k,m] = np.nanmean(out_rHetAer[-nn_outputforaverage::])
        fin_rO2C[k,m] = np.nanmean(out_rO2C[-nn_outputforaverage::])
        fin_r1Den[k,m] = np.nanmean(out_r1Den[-nn_outputforaverage::])
        fin_r2Den[k,m] = np.nanmean(out_r2Den[-nn_outputforaverage::])
        fin_r3Den[k,m] = np.nanmean(out_r3Den[-nn_outputforaverage::])
        fin_r4Den[k,m] = np.nanmean(out_r4Den[-nn_outputforaverage::]) 
        fin_r5Den[k,m] = np.nanmean(out_r5Den[-nn_outputforaverage::]) 
        fin_r6Den[k,m] = np.nanmean(out_r6Den[-nn_outputforaverage::]) 
        #fin_r7Den[k,m] = np.nanmean(out_r7Den[-nn_outputforaverage::])
        # #add copi (6 dens)
        # fin_r1DenC[k,m] = np.nanmean(out_r1DenC[-200::])
        # fin_r2DenC[k,m] = np.nanmean(out_r2DenC[-200::])
        # fin_r3DenC[k,m] = np.nanmean(out_r3DenC[-200::])
        # fin_r4DenC[k,m] = np.nanmean(out_r4DenC[-200::]) 
        # fin_r5DenC[k,m] = np.nanmean(out_r5DenC[-200::]) 
        # fin_r6DenC[k,m] = np.nanmean(out_r6DenC[-200::]) 
        #end
        fin_rAOO[k,m] = np.nanmean(out_rAOO[-nn_outputforaverage::])
        fin_rNOO[k,m] = np.nanmean(out_rNOO[-nn_outputforaverage::])
        fin_rAOX[k,m] = np.nanmean(out_rAOX[-nn_outputforaverage::])
        
        fin_bNOOan[k,m] = np.nanmean(out_bNOOan[-nn_outputforaverage::])
        fin_uNOOan[k,m] = np.nanmean(out_uNOOan[-nn_outputforaverage::])
        fin_rNOOan[k,m] = np.nanmean(out_rNOOan[-nn_outputforaverage::])
        
        fin_rN2Oammonia[k,m] = np.nanmean(out_rN2Oammonia[-nn_outputforaverage::])
        if len(outputd1)*len(outputd2) < 3:
            print("mean nitrite", fin_NO2[0])
            print("mean O2", fin_O2[0])
            
# 6. check conservation of mass if dilution rate is set to zero
if dil == 0.0:
    end_N = fin_Sd + fin_Sp + fin_NO3 + fin_NO2 + fin_NH4 + fin_N2 + fin_N2O + \
            fin_bHet + fin_bFac + fin_b1Den + fin_b2Den + fin_b3Den + fin_b4Den + fin_b5Den + fin_b6Den + fin_bAOO + fin_bNOO + fin_bAOX + \
            fin_bHetC + fin_b1DenC + fin_b2DenC + fin_b3DenC + fin_b4DenC + fin_b5DenC + fin_b6DenC
    ini_N = Sd0_exp + in_Sp + in_NO3 + in_NO2 + in_NH4 + \
            in_bHet + in_bFac + in_b1Den + in_b2Den + in_b3Den + in_b4Den + in_b5Den + in_b6Den + in_bAOO + in_bNOO + in_bAOX + \
            in_bHetC + in_b1DenC + in_b2DenC + in_b3DenC + in_b4DenC + in_b5DenC + in_b6DenC
    for k in np.arange(len(Sd0_exp)):
        for m in np.arange(len(O20_exp)):
            print(" Checking conservation of N mass ")
            print(" Initial Nitrogen =", ini_N[k])
            print(" Final Nitrogen =", end_N[k,m])
            
# delete results only save fin (average)
del results
del out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_N2O
del out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den, out_b7Den, out_bAOO, out_bNOO, out_bNOOan, out_bAOX
del out_bHetC, out_b1DenC, out_b2DenC, out_b3DenC, out_b4DenC, out_b5DenC, out_b6DenC
del out_uHet, out_uFac, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den, out_u7Den, out_uAOO, out_uNOO, out_uNOOan, out_uAOX
del out_uHetC, out_u1DenC, out_u2DenC, out_u3DenC, out_u4DenC, out_u5DenC, out_u6DenC,
del out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rNOOan, out_rAOX
#del out_r1DenC, out_r2DenC, out_r3DenC, out_r4DenC, out_r5DenC, out_r6DenC




#%% Plotting results
"copio-oligo model figures"


#%% Fig 2_(Biomass & Copiotrophy)
fstic = 16
fslab = 18
fsleg = 10.5
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(6,7)) #figsize=(6,10)
gs = GridSpec(2, 1)

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('')

OMx = (xpulse_Sd/xpulse_int + Sd0_exp * dil) / (30 * dil)

linewidtholigo = 3
linewidthcopio = 3
linealpha = 0.5

### plot together
plt.plot(OMx, fin_b1DenC, '-', color='firebrick', label='C NO$_3$$^-$→NO$_2$$^-$', linewidth=linewidthcopio, alpha = linealpha)
plt.plot(OMx, fin_b6DenC, '--', color='firebrick', label='C NO$_3$$^-$→N$_2$O', linewidth=linewidthcopio, alpha = linealpha)
plt.plot(OMx, fin_b3DenC, ':', color='firebrick', label='C NO$_3$$^-$→N$_2$', linewidth=linewidthcopio, alpha = linealpha)
plt.plot(OMx, fin_b4DenC, '-', color='goldenrod', label='C NO$_2$$^-$→N$_2$O', linewidth=linewidthcopio, alpha = linealpha)
plt.plot(OMx, fin_b2DenC, '--', color='goldenrod', label='C NO$_2$$^-$→N$_2$', linewidth=linewidthcopio, alpha = linealpha)
plt.plot(OMx, fin_b5DenC, '-', color='royalblue', label='C N$_2$O→N$_2$', linewidth=linewidthcopio, alpha = linealpha) 

plt.plot(OMx, fin_b1Den, '-', color='firebrick', label='O NO$_3$$^-$→NO$_2$$^-$', linewidth=linewidtholigo)
plt.plot(OMx, fin_b6Den, '--', color='firebrick', label='O NO$_3$$^-$→N$_2$O', linewidth=linewidtholigo)
plt.plot(OMx, fin_b3Den, ':', color='firebrick', label='O NO$_3$$^-$→N$_2$', linewidth=linewidtholigo)
plt.plot(OMx, fin_b5Den, '-', color='royalblue', label='O N$_2$O→N$_2$', linewidth=linewidtholigo) 
plt.plot(OMx, fin_b4Den, '-', color='goldenrod', label='O NO$_2$$^-$→N$_2$O', linewidth=linewidtholigo)
plt.plot(OMx, fin_b2Den, '--', color='goldenrod', label='O NO$_2$$^-$→N$_2$', linewidth=linewidtholigo)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=1)
ax1.set_ylabel('Biomass (µM-N)', fontsize=fslab) 
ax1.set_ylim(bottom=0)

# creat subplots
ax2 = plt.subplot(gs[1,0])
ax2.set_title('')
def plot_percent(ax, x, numerator, denominator, threshold = 1e-10, **plot_kwargs):
    """
    Plot percentage of numerator / (numerator + denominator) * 100,
    masking any values where total < threshold.
    """
    total = numerator + denominator
    percent = np.where(total < threshold, np.nan, numerator / total * 100)
    ax.plot(x, percent, **plot_kwargs)

plot_percent(ax2, OMx, fin_b1DenC+fin_b3DenC+fin_b6DenC, fin_b1Den+fin_b3Den+fin_b6Den, color='firebrick', linestyle='-', label='NO$_3$$^-$ reducers', linewidth=linewidtholigo, alpha = 1)
plot_percent(ax2, OMx, fin_b1DenC+fin_b2DenC+fin_b3DenC+fin_b4DenC+fin_b5DenC+fin_b6DenC, fin_b1Den+fin_b2Den+fin_b3Den+fin_b4Den+fin_b5Den+fin_b6Den, color='k', linestyle='-', label='All', linewidth=linewidtholigo, alpha = linealpha)

ax2.set_ylabel('Percent copiotrophy (%)', fontsize=fslab)
ax2.set_xlabel('OM:NO$_3$$^-$ supply', fontsize=fslab)
ax2.set_ylim(bottom=0)

ax1.tick_params(labelbottom=False, labelsize=fstic)
ax2.tick_params(labelsize=fstic)

xlowerlimit = 0
xupperlimit = 0.365
xtickdiv = 0.1
for ax in [ax1, ax2]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

plt.legend(loc='upper right', fontsize=12) 

plt.tight_layout()

#%%
fig.savefig('Fig2_alt.png', dpi=300) 


#%% Fig 4a Model-data_bar plots of copiotrophy for each functional type 

import numpy as np
import matplotlib.pyplot as plt

# 
fstic = 16
fslab = 18
fsleg = 10.5

# function
def compute_percent(numerator, denominator, threshold=1e-10):
    total = numerator + denominator
    percent = np.where(total < threshold, np.nan, numerator / total * 100)
    return percent

# Functional types and labels
types = {
    'NO$_3$$^-$→NO$_2$$^-$': (fin_b1DenC, fin_b1Den),
    'NO$_3$$^-$→N$_2$O':     (fin_b6DenC, fin_b6Den),
    'NO$_3$$^-$→N$_2$':      (fin_b3DenC, fin_b3Den),  # This one will get shaded range
    'NO$_2$$^-$→N$_2$O':     (fin_b4DenC, fin_b4Den),
    'NO$_2$$^-$→N$_2$':      (fin_b2DenC, fin_b2Den),
    'N$_2$O→N$_2$':          (fin_b5DenC, fin_b5Den),
}
colors = {
    'NO$_3$$^-$→NO$_2$$^-$': 'firebrick',
    'NO$_3$$^-$→N$_2$O':     'firebrick',
    'NO$_3$$^-$→N$_2$':      'firebrick',
    'NO$_2$$^-$→N$_2$O':     'goldenrod',
    'NO$_2$$^-$→N$_2$':      'goldenrod',
    'N$_2$O→N$_2$':          'royalblue',
}

# mean percent copiotrophy
mean_percent_dict = {}
for label, (numer, denom) in types.items():
    percent = compute_percent(numer, denom)
    mean_percent = np.nanmean(percent)
    mean_percent_dict[label] = mean_percent

# Sensitivity analysis for complete denitrifiers
percent_b3 = compute_percent(fin_b3DenC, fin_b3Den)
thresholds = np.linspace(0.05, 0.35, 100)
mean_range = []

for t in thresholds:
    mask = (OMx <= t) & ~np.isnan(percent_b3)
    if np.sum(mask) > 0:
        mean_range.append(np.mean(percent_b3[mask]))
    else:
        mean_range.append(np.nan)

min_val = np.nanmin(mean_range)
max_val = np.nanmax(mean_range)

#Plot
labels = list(mean_percent_dict.keys())
mean_vals = list(mean_percent_dict.values())
bar_colors = [colors[label] for label in labels]

fig, ax = plt.subplots(figsize=(5.6, 4.8))
bars = ax.bar(labels, mean_vals, color=bar_colors, edgecolor='black')

# Add shaded range for NO₃⁻→N₂
highlight_label = 'NO$_3$$^-$→N$_2$'
idx = labels.index(highlight_label)
x_center = idx
bar_width = 0.8

# Draw the shaded uncertainty band
ax.fill_betweenx(
    y=[min_val, max_val],
    x1=x_center - bar_width / 2,
    x2=x_center + bar_width / 2,
    color='firebrick',
    alpha=0.3,
    zorder=0,
    label='Uncertainty range\n(NO$_3$$^-$→N$_2$)'
)

# Labels and aesthetics
ax.set_ylabel('Mean percent copiotrophy (%)', fontsize=fslab)
ax.set_xlabel('', fontsize=fslab)
ax.set_xticks(range(len(labels)))
ax.tick_params(axis='x', length=0)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fstic)
ax.tick_params(axis='y', labelsize=fstic)
ax.set_ylim(0, 100)
ax.set_ylabel('Mean percent copiotrophy (%)', fontsize=fslab, weight='bold')

plt.tight_layout()
plt.show()

#%%
os.chdir("/Users/xinsun/Dropbox/!Carnegie/!Research/!Manuscript_Den&SNM&N2O/3.CopioOligo-lifestyle")
fig.savefig('Fig4a.png', dpi=300) 


#%% FigS3
fstic = 16
fslab = 18
fsleg = 10.5
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(5,10)) #figsize=(5,10)
gs = GridSpec(3, 1)

OMx = (xpulse_Sd/xpulse_int + Sd0_exp * dil) / (30 * dil)

linewidtholigo = 3
linewidthcopio = 3

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('Total', fontsize=fslab)
plt.plot(OMx, fin_b1Den + fin_b1DenC, '-', color='firebrick', label='NO$_3$$^-$→NO$_2$$^-$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b6Den + fin_b6DenC, '--', color='firebrick', label='NO$_3$$^-$→N$_2$O', linewidth=linewidthcopio)
plt.plot(OMx, fin_b3Den + fin_b3DenC, ':', color='firebrick', label='NO$_3$$^-$→N$_2$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b4Den + fin_b4DenC, '-', color='goldenrod', label='NO$_2$$^-$→N$_2$O', linewidth=linewidthcopio)
plt.plot(OMx, fin_b2Den + fin_b2DenC, '--', color='goldenrod', label='NO$_2$$^-$→N$_2$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b5Den + fin_b5DenC, '-', color='royalblue', label='N$_2$O→N$_2$', linewidth=linewidthcopio) 

#plt.legend(loc='lower right', fontsize=9) 
ax1.set_ylabel('Biomass (µM-N)', fontsize=fslab)


ax2 = plt.subplot(gs[1,0])
ax2.set_title('Oligotrophs', fontsize=fslab)
plt.plot(OMx, fin_b1Den, '-', color='firebrick', label='NO$_3$$^-$→NO$_2$$^-$', linewidth=linewidtholigo)
plt.plot(OMx, fin_b6Den, '--', color='firebrick', label='NO$_3$$^-$→N$_2$O', linewidth=linewidtholigo)
plt.plot(OMx, fin_b3Den, ':', color='firebrick', label='NO$_3$$^-$→N$_2$', linewidth=linewidtholigo)
plt.plot(OMx, fin_b4Den, '-', color='goldenrod', label='NO$_2$$^-$→N$_2$O', linewidth=linewidtholigo) # hard to tell but are there!
plt.plot(OMx, fin_b2Den, '--', color='goldenrod', label='NO$_2$$^-$→N$_2$', linewidth=linewidtholigo)
plt.plot(OMx, fin_b5Den, '-', color='royalblue', label='N$_2$O→N$_2$', linewidth=linewidtholigo) 
#plt.yscale('log')
ax2.set_ylabel('Biomass (µM-N)', fontsize=fslab)


ax3 = plt.subplot(gs[2,0])
ax3.set_title('Copiotrophs', fontsize=fslab)
plt.plot(OMx, fin_b1DenC, '-', color='firebrick', label='NO$_3$$^-$→NO$_2$$^-$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b6DenC, '--', color='firebrick', label='NO$_3$$^-$→N$_2$O', linewidth=linewidthcopio)
plt.plot(OMx, fin_b3DenC, ':', color='firebrick', label='NO$_3$$^-$→N$_2$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b4DenC, '-', color='goldenrod', label='NO$_2$$^-$→N$_2$O', linewidth=linewidthcopio)
plt.plot(OMx, fin_b2DenC, '--', color='goldenrod', label='NO$_2$$^-$→N$_2$', linewidth=linewidthcopio)
plt.plot(OMx, fin_b5DenC, '-', color='royalblue', label='N$_2$O→N$_2$', linewidth=linewidthcopio) 
ax3.set_ylabel('Biomass (µM-N)', fontsize=fslab)
ax3.set_xlabel('OM:NO$_3$$^-$ supply with pulses', fontsize=fslab)

ax1.tick_params(labelbottom=False, labelsize=fstic)
ax2.tick_params(labelbottom=False, labelsize=fstic)
ax3.tick_params(labelsize=fstic)

xlowerlimit = 0
xupperlimit = 0.365 
xtickdiv = 0.1
for ax in [ax1, ax2, ax3]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

ax1.set_ylim([0.0, 1.0])
ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
ax2.set_ylim([0.0, 1.0])
ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
ax3.set_ylim([0.0, 1.0])
ax3.set_yticks(np.arange(0.0, 1.1, 0.2))

plt.tight_layout()

#%% Save the plot
os.chdir("/Users/xinsun/Dropbox/!Carnegie/!Research/!Manuscript_Den&SNM&N2O/3.CopioOligo-lifestyle")
fig.savefig('FigS3_pulse50days.png', dpi=300) 


#%% Copio-oligo_FigS1a_(OM* and N* with oligotrophs-den only)

fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(4,5.5))

gs = GridSpec(2, 1)


ax1 = plt.subplot(gs[0,0])
#ax1.set_title('OM*', fontsize=fslab)
## Create a list of values for the y axis
y_values = [OM_star_den1, OM_star_den6, OM_star_den3, OM_star_den4, OM_star_den2, OM_star_den5]
## Create a list of categories for the x axis
x_categories = ['NO$_3$$^-$→NO$_2$$^-$', 'NO$_3$$^-$→N$_2$O','NO$_3$$^-$→N$_2$', 'NO$_2$$^-$→N$_2$O', 'NO$_2$$^-$→N$_2$', 'N$_2$O→N$_2$']
## Create a list of colors for the bars
colors = ['firebrick','firebrick','firebrick', 'goldenrod','goldenrod','royalblue']

## Create a bar plot
barOM = plt.bar(x_categories, y_values, 
        color='white',
        edgecolor=colors,
        linewidth=2)

# Define different line styles for each bar
line_styles = ['-', '--', ':', '-', '--', '-']

# Apply different line styles to each bar
for bar, linestyle in zip(barOM, line_styles):
    bar.set_linestyle(linestyle)
plt.show()

## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
#plt.xticks(rotation=90, fontsize=fslab)
ax1.set_ylim(ymin = 0, ymax = 0.032)
ax1.set_yticks(np.arange(0, 0.032, 0.01))
plt.ylabel('OM* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)


ax2 = plt.subplot(gs[1,0])
y_values2 = [nitrate_star_den1, nitrate_star_den6, nitrate_star_den3, nitrite_star_den4, nitrite_star_den2, N2O_star_den5*1e-3]
## Create a list of colors for the bars
colors = ['firebrick','firebrick','firebrick', 'goldenrod','goldenrod','royalblue']
## Create a bar plot
barN = plt.bar(x_categories, y_values2, 
        color='white',
        edgecolor=colors,
        linewidth=2)

# Define different line styles for each bar
line_styles = ['-', '--', ':', '-', '--', '-']

# Apply different line styles to each bar
for bar, linestyle in zip(barN, line_styles):
    bar.set_linestyle(linestyle)
plt.show()

## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
plt.xticks(rotation=90, fontsize=fslab)

plt.ylabel('N* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)

ax1.tick_params(labelbottom=False)

plt.tight_layout()

#%% Copio-oligo_FigS1b_(OM* and N* with Copiotrophs-den only)

fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(4,5.5))

gs = GridSpec(2, 1)


ax1 = plt.subplot(gs[0,0])
#ax1.set_title('OM*', fontsize=fslab)
## Create a list of values for the y axis
y_values = [OM_star_den1C, OM_star_den6C, OM_star_den3C, OM_star_den4C, OM_star_den2C, OM_star_den5C]
## Create a list of categories for the x axis
x_categories = ['NO$_3$$^-$→NO$_2$$^-$', 'NO$_3$$^-$→N$_2$O','NO$_3$$^-$→N$_2$', 'NO$_2$$^-$→N$_2$O', 'NO$_2$$^-$→N$_2$', 'N$_2$O→N$_2$']
## Create a list of colors for the bars
colors = ['firebrick','firebrick','firebrick', 'goldenrod','goldenrod','royalblue']

## Create a bar plot
barOM = plt.bar(x_categories, y_values, 
        color='white',
        edgecolor=colors,
        linewidth=2)

# Define different line styles for each bar
line_styles = ['-', '--', ':', '-', '--', '-']

# Apply different line styles to each bar
for bar, linestyle in zip(barOM, line_styles):
    bar.set_linestyle(linestyle)
plt.show()

## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
#plt.xticks(rotation=90, fontsize=fslab)
ax1.set_ylim(ymin = 0, ymax = 0.152)
ax1.set_yticks(np.arange(0, 0.152, 0.05))
plt.ylabel('OM* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)


ax2 = plt.subplot(gs[1,0])
y_values2 = [nitrate_star_den1C, nitrate_star_den6C, nitrate_star_den3C, nitrite_star_den4C, nitrite_star_den2C, N2O_star_den5C*1e-3]
## Create a list of colors for the bars
colors = ['firebrick','firebrick','firebrick', 'goldenrod','goldenrod','royalblue']
## Create a bar plot
barN = plt.bar(x_categories, y_values2, 
        color='white',
        edgecolor=colors,
        linewidth=2)

# Define different line styles for each bar
line_styles = ['-', '--', ':', '-', '--', '-']

# Apply different line styles to each bar
for bar, linestyle in zip(barN, line_styles):
    bar.set_linestyle(linestyle)
plt.show()

## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
plt.xticks(rotation=90, fontsize=fslab)
ax2.set_yticks(np.arange(0, 1.52, 0.5))

plt.ylabel('N* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)

ax1.tick_params(labelbottom=False)

plt.tight_layout()

#%% Save the plot
os.chdir("/Users/xinsun/Library/CloudStorage/Dropbox/!Carnegie/!Research/!Manuscript_Den&SNM&N2O/3.CopioOligo-lifestyle")
fig.savefig('FigS1_R*.png', dpi=300)

#%% save the ,model output
os.chdir("/Users/xinsun/Dropbox/!Carnegie/!Research/ChemostatModel_ModularDenitrification/Output/output=dt-Pulse22Oxy10withaerNOO-5e4_withCopioCN_Oxy0_pulse5days") 

fname = 'OMpulse_P0.16'


np.savetxt(fname+'_pulse.txt', xpulse_Sd, delimiter='\t')
np.savetxt(fname+'_O2supply.txt', O20_exp, delimiter='\t')

np.savetxt(fname+'_O2.txt', fin_O2, delimiter='\t')
np.savetxt(fname+'_N2.txt', fin_N2, delimiter='\t')
np.savetxt(fname+'_N2O.txt', fin_N2O, delimiter='\t')
np.savetxt(fname+'_NO3.txt', fin_NO3, delimiter='\t')
np.savetxt(fname+'_NO2.txt', fin_NO2, delimiter='\t')
np.savetxt(fname+'_NH4.txt', fin_NH4, delimiter='\t')
np.savetxt(fname+'_OM.txt', fin_Sd, delimiter='\t')

np.savetxt(fname+'_bHet.txt', fin_bHet, delimiter='\t')
np.savetxt(fname+'_b1Den.txt', fin_b1Den, delimiter='\t')
np.savetxt(fname+'_b2Den.txt', fin_b2Den, delimiter='\t')
np.savetxt(fname+'_b3Den.txt', fin_b3Den, delimiter='\t')
np.savetxt(fname+'_b4Den.txt', fin_b4Den, delimiter='\t')
np.savetxt(fname+'_b5Den.txt', fin_b5Den, delimiter='\t')
np.savetxt(fname+'_b6Den.txt', fin_b6Den, delimiter='\t')
np.savetxt(fname+'_b7Den.txt', fin_b7Den, delimiter='\t')
np.savetxt(fname+'_bAOO.txt', fin_bAOO, delimiter='\t')
np.savetxt(fname+'_bNOO.txt', fin_bNOO, delimiter='\t')
np.savetxt(fname+'_bNOOan.txt', fin_bNOOan, delimiter='\t')
np.savetxt(fname+'_bAOX.txt', fin_bAOX, delimiter='\t')

np.savetxt(fname+'_uHet.txt', fin_uHet, delimiter='\t')
np.savetxt(fname+'_u1Den.txt', fin_u1Den, delimiter='\t')
np.savetxt(fname+'_u2Den.txt', fin_u2Den, delimiter='\t')
np.savetxt(fname+'_u3Den.txt', fin_u3Den, delimiter='\t')
np.savetxt(fname+'_u4Den.txt', fin_u4Den, delimiter='\t')
np.savetxt(fname+'_u5Den.txt', fin_u5Den, delimiter='\t')
np.savetxt(fname+'_u6Den.txt', fin_u6Den, delimiter='\t')
np.savetxt(fname+'_u7Den.txt', fin_u7Den, delimiter='\t')
np.savetxt(fname+'_uAOO.txt', fin_uAOO, delimiter='\t')
np.savetxt(fname+'_uNOO.txt', fin_uNOO, delimiter='\t')
np.savetxt(fname+'_uAOX.txt', fin_uAOX, delimiter='\t')

np.savetxt(fname+'_facaer.txt', fin_facaer, delimiter='\t')
np.savetxt(fname+'_rHet.txt', fin_rHet, delimiter='\t')
np.savetxt(fname+'_rHetAer.txt', fin_rHetAer, delimiter='\t')
np.savetxt(fname+'_r1Den.txt', fin_r1Den, delimiter='\t')
np.savetxt(fname+'_r2Den.txt', fin_r2Den, delimiter='\t')
np.savetxt(fname+'_r3Den.txt', fin_r3Den, delimiter='\t')
np.savetxt(fname+'_r4Den.txt', fin_r4Den, delimiter='\t')
np.savetxt(fname+'_r5Den.txt', fin_r5Den, delimiter='\t')
np.savetxt(fname+'_r6Den.txt', fin_r6Den, delimiter='\t')
np.savetxt(fname+'_rAOO.txt', fin_rAOO, delimiter='\t')
np.savetxt(fname+'_rNOO.txt', fin_rNOO, delimiter='\t')
np.savetxt(fname+'_rAOX.txt', fin_rAOX, delimiter='\t')
np.savetxt(fname+'_rO2C.txt', fin_rO2C, delimiter='\t')

np.savetxt(fname+'_rN2Oammonia.txt', fin_rN2Oammonia, delimiter='\t')

np.savetxt(fname+'_bHetC.txt', fin_bHetC, delimiter='\t')
np.savetxt(fname+'_b1DenC.txt', fin_b1DenC, delimiter='\t')
np.savetxt(fname+'_b2DenC.txt', fin_b2DenC, delimiter='\t')
np.savetxt(fname+'_b3DenC.txt', fin_b3DenC, delimiter='\t')
np.savetxt(fname+'_b4DenC.txt', fin_b4DenC, delimiter='\t')
np.savetxt(fname+'_b5DenC.txt', fin_b5DenC, delimiter='\t')
np.savetxt(fname+'_b6DenC.txt', fin_b6DenC, delimiter='\t')

np.savetxt(fname+'_uHetC.txt', fin_uHetC, delimiter='\t')
np.savetxt(fname+'_u1DenC.txt', fin_u1DenC, delimiter='\t')
np.savetxt(fname+'_u2DenC.txt', fin_u2DenC, delimiter='\t')
np.savetxt(fname+'_u3DenC.txt', fin_u3DenC, delimiter='\t')
np.savetxt(fname+'_u4DenC.txt', fin_u4DenC, delimiter='\t')
np.savetxt(fname+'_u5DenC.txt', fin_u5DenC, delimiter='\t')
np.savetxt(fname+'_u6DenC.txt', fin_u6DenC, delimiter='\t')
