# -*- coding: utf-8 -*-
"""
March 2023 

Purpose
-------
    N2O and modular denitrification projects Xin Sun
    Modified traits based on Zakem et al. 2019 ISME
    yields of denitrifiers depend on Gibbs free energy,
    copiotrophs and oligotrophs are included

@authors: Xin Sun, Pearse Buchanan and Emily Zakem
"""

import numpy as np
from diffusive_gas_coefficient import po_coef
from diffusive_gas_coefficient import pn2o_coef
from yield_from_stoichiometry import yield_stoich
from GibbsEnergyYield_Xin import calc_y
P = 0.16 #0.16-0.22 (>0.2254 P making the N* not all opposite with OM*) #0.14912600274046617 (P must be < 0.428, for all y and OM* to be larger than 0. 0.5, if = 0.5, 1-2P= 0, no e- for 2 step den)
#When P = 0.14912600274046617, y_n1Den (NO3->NO2) = y_n6Den (NO3->N2O) = 0.14965365800572378
#y_n3Den = 0.1419376500321851 < y_n1Den

K_n_all = 2 # 1 same; OR other numbber (2) diff

### 1) max growth rates (per day)
mumax_Het = 0.3 #1#for testing #0.3      # Now 0.3; Previous value 0.5; 0.5 based on Rappé et al., 2002; 0.1 based on Kirchman 2016 
mumax_AOO = 0.25    # Previous value 0.5; Kitzinger et al., 2020
mumax_NOO = 1.0     # Now 1.0; Previous value 0.96; Kitzinger et al., 2020
mumax_AOX = 0.1     # Previous value 0.2; This is too low 0.048 based on Oshiki et al., 2016 (Scalindua in Table 1); too close to dilution rate gonna be washed out


### 2) diffusive oxygen and n2o requirements based on cell diameters and carbon contents

# cell volumes (um^3)
vol_aer = 0.05  # based on SAR11 (Giovannoni 2017 Ann. Rev. Marine Science)
vol_den = vol_aer
vol_aoo = np.pi * (0.2*0.5)**2 * 0.8    # [rods] N. maritimus SCM1 in Table S1 from Hatzenpichler 2012 App. Envir. Microbiology
vol_noo = np.pi * (0.3*0.5)**2 * 3      # [rods] based on average cell diameter and length of Nitrospina gracilis (Spieck et al. 2014 Sys. Appl. Microbiology)
vol_aox = 4/3 * np.pi * (0.8*0.5)**3    # [cocci] diameter of 0.8 um based on Candidatus Scalindua (Wu et al. 2020 Water Science and Tech.)

# cell sizes (um), where vol = 4/3 * pi * r^3
diam_aer = ( 3 * vol_aer / (4 * np.pi) )**(1./3)*2
diam_den = ( 3 * vol_den / (4 * np.pi) )**(1./3)*2
diam_aoo = ( 3 * vol_aoo / (4 * np.pi) )**(1./3)*2
diam_noo = ( 3 * vol_noo / (4 * np.pi) )**(1./3)*2
diam_aox = ( 3 * vol_aox / (4 * np.pi) )**(1./3)*2

# cellular C:N of microbes
CN_aer = 5.0   # Zimmerman et al. 2014
CN_den = 5.0
CN_aoo = 5.0    
CN_noo = 5.0    
CN_aox = 5.0    

# g carbon per cell (assumes 0.1 g DW/WW for all microbial types as per communication with Marc Strous)
Ccell_aer = 0.1 * (12*CN_aer / (12*CN_aer + 7 + 16*2 + 14)) / (1e12 / vol_aer)
Ccell_den = 0.1 * (12*CN_den / (12*CN_den + 7 + 16*2 + 14)) / (1e12 / vol_den)
Ccell_aoo = 0.1 * (12*CN_aoo / (12*CN_aoo + 7 + 16*2 + 14)) / (1e12 / vol_aoo)
Ccell_noo = 0.1 * (12*CN_noo / (12*CN_noo + 7 + 16*2 + 14)) / (1e12 / vol_noo)
Ccell_aox = 0.1 * (12*CN_aox / (12*CN_aox + 8.7 + 16*1.55 + 14)) / (1e12 / vol_aox)

# cell quotas (mol C / um^3)
Qc_aer = Ccell_aer / vol_aer / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_den = Ccell_den / vol_den / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_aoo = Ccell_aoo / vol_aoo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_noo = Ccell_noo / vol_noo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_aox = Ccell_aox / vol_aox / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019

# diffusive oxygen and n2o coefficient
dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
dc_n2o = dc * 1.0049    # m^2/day (this is necessary to calculate N2O star in call_moel_XinanoxicODZ’s ‘#%% calculate R*-stars for all microbes’)

po_aer = po_coef(diam_aer, Qc_aer, CN_aer)
po_aoo = po_coef(diam_aoo, Qc_aoo, CN_aoo)
po_noo = po_coef(diam_noo, Qc_noo, CN_noo)

po_den = po_coef(diam_den, Qc_den, CN_den) # may not need these two
po_aox = po_coef(diam_aox, Qc_aox, CN_aox) # may not need these two

pn2o_den = pn2o_coef(diam_den, Qc_den, CN_den)


### 3) Yields (y), products (e) and maximum uptake rate (Vmax)
###     Vmax = ( mu_max * Quota ) / yield 
###     we remove the need for the Quota term by considering everything in mols N per mol N-biomass, such that
###     Vmax = mu_max / yield (units of mol N per mol N-biomass per day)

### Delta G depended organic matter yield for heterotrophs (aerobes and denitrifiers)
##### THERMODYNAMIC DEFINITIONS
##### authors: Emily Zakem and Xin Sun

## Xin: define parameters
T = 25 # temperature (degree C)
pH = 7.5 
R = 8.3145 #J/(mol*K) Ideal gas constant
T = T + 273.15 #Converting temperature above to Kelvin
H = 10**(-pH) #Converting pH to concentration in units of mol/L

## Free energy of formation for relevant compounds and phases all values
## from (Amend & Shock 2001) at 25 degrees C
## EJZ: at pH = 0, not 7 (diff than R+McC chart)
DGf_NO3 = -110.91 * 1e3 #J/mol (aq) [NO3-]
DGf_NO2 = -32.22  * 1e3 #J/mol (aq) [NO2-]
DGf_H2O = -237.18 * 1e3 #J/mol (l)
DGf_N2O =  113.38 * 1e3 #J/mol (aq)
DGf_H   =    0    * 1e3 #J/mol (aq) [H+]
DGf_NO  =  102.06 * 1e3 #J/mol (aq)
DGf_N2  =   18.18 * 1e3 #J/mol (aq)
DGf_O2  =   16.54 * 1e3 #J/mol (aq) # Xin: add O2 in the list

## Calculating the standard free energy for the half reactions
## Xin: 1, 2, 3, 4 is NO3-->NO2-->NO-->N2O-->N2, ignore reactions producing or consuming NO below
# unit of DGo below is J/mol
DGheto = 0.5*DGf_H2O - 1/4*DGf_O2 - DGf_H # Xin: add a set for aerobic heterotrophs 
DG1o = 0.5*DGf_NO2 + 0.5*DGf_H2O - 0.5*DGf_NO3 - DGf_H 
DG4o = 0.5*DGf_N2 + 0.5*DGf_H2O - 0.5*DGf_N2O - DGf_H 
DG123o = 1/8*DGf_N2O + 5/8*DGf_H2O - 1/4*DGf_NO3 - 5/4*DGf_H 
DG1234o = 1/10*DGf_N2 + 3/5*DGf_H2O - 1/5*DGf_NO3 - 6/5*DGf_H 
DG23o = 1/4*DGf_N2O + 3/4*DGf_H2O - 1/2*DGf_NO2 - 3/2*DGf_H 
DG234o = 1/6*DGf_N2 + 2/3*DGf_H2O - 1/3*DGf_NO2 - 4/3*DGf_H 

#avg concentrations (to initiate -- but don't want this to determine answer):
NO3avg = 30 * 1e-6 #mol/L 
NO2avg = 1e-6 #mol/L 
N2Oavg = 1e-8 #mol/L 
N2avg = 1e-4 #mol/L 
O2avg = 1e-6 # Xin: add this for aerobic heterotrophs

#per e- (In Julia, default based of log is e)
DGhet = DGheto + R*T*np.log(1/O2avg**(1/4)/H) # Xin: add a set for aerobic heterotrophs
DG1 = DG1o + R*T*np.log(NO2avg**0.5/NO3avg**0.5/H) 
DG4 = DG4o + R*T*np.log(N2avg**0.5/N2Oavg**0.5/H)
DG123  = DG123o  + R*T*np.log(N2Oavg**(1/8)/NO3avg**(1/4)/H**(5/4)) 
DG1234 = DG1234o + R*T*np.log(N2avg**(1/10)/NO3avg**(1/5)/H**(6/5)) 
DG23  = DG23o  + R*T*np.log(N2Oavg**(1/4)/NO2avg**(1/2)/H**(3/2)) 
DG234 = DG234o + R*T*np.log(N2avg**(1/6)/NO2avg**(1/3)/H**(4/3)) 

## Gibbs rxn energy for oxidation of organic matter
## (here, estimated for avg marine OM:
om_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cd=6.6; Hd=10.9; Od=2.6; Nd=1 # Based on (Anderson 1995)
## denominator for decomposition to nh4:
dD=4*Cd+Hd-2*Od-3*Nd;
DGom = om_energy*(Cd*12+Hd+Od*16+Nd*14)/dD

## energy to form biomass b from pyruvate -- R&McC:
b_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cb=5; Hb=7; Ob=2; Nb=1 # Based on (Zimmerman et al. 2014)
dB=4*Cb+Hb-2*Ob-3*Nb
DGpb = b_energy*(Cb*12+Hb+Ob*16+Nb*14)/dB #energy to synthesize cells from pyruvate

## efficiency of e transfer:
ep = 0.6

## total energy needed for cell synthesis from given OM, including conv to pyruvate 
DGp = 35.09*1e3 - DGom #e to convert C to pyruvate. CHECK this 35.09 -- units?
#if Gp>=0; n=1; end;if Gp<0; n=-1; end 
n=1 #assume for now DGp > 0 (typical)
DGs = DGp/ep**n + DGpb/ep # total energy needed for synthesis. add uptake cost to this? 
## total energy needed for cell synthesis from given OM:
## energy needed to convert OM to pyruvate + energy needed to convert pyruvate to biomass
##### Finish defining THERMODYNAMICs
DGs = 300*1e3 # reset DGs to get the yield of aerobic hetero close to obs (0.2~0.3)

dO = 29.1   # assumes constant stoichiometry of all sinking organic matter of C6.6-H10.9-O2.6-N using equation: d = 4C + H - 2O -3N
dB = 20.0   # assumes constant stoichiometry of all heterotrophic bacteria of C5-H7-O2-N based on Zimmerman et al., 2014

# solve for the yield of heterotrophic bacteria using approach of Sinsabaugh et al. 2013
Y_max = 0.6     # (will not need this with GibbsFreeEnergy defined yield) maximum possible growth efficiency measured in the field
B_CN = 5.0      # C:N of bacterial biomass based on Zimmerman et al. 2014
#B_CN = 4.5      # C:N of bacterial biomass (White et al. 2019)
OM_CN = 6.6     # C:N of labile dissolved organic matter (Letscher et al. 2015)
K_CN = 0.5      # C:N half-saturation coefficient (Sinsabaugh & Follstad Shah 2012 - Ecoenzymatic Stoichiometry and Ecological Theory)
EEA_CN = 1.123  # relative rate of enzymatic processing of complex C and complex N molecules into simple precursors for biosynthesis (Sinsabaugh & Follstad Shah 2012)

# aerobic heterotrophy 
f_oHet = calc_y(DGhet, DGom, ep, DGs) # old: y_oHet * dB/dO  # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oHet =  f_oHet * dO/dB # old: yield_stoich(Y_max, B_CN, OM_CN, K_CN, EEA_CN) / B_CN * OM_CN  # OM_CN / B_CN converts to mol BioN per mol OrgN 
y_oO2 = (f_oHet/dB) / ((1.0-f_oHet)/4.0)  # yield of biomass per unit oxygen reduced
VmaxS = mumax_Het / y_oHet  # max mol Org N consum. rate per (mol BioN) per day

# Copiotroph
VmaxSC = VmaxS * 4 # to make sure the V/k_copiotroph/v/k_oligo is the same between C and N.

## artifical value to set below DIN Vmax really high
## VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_4Den and VmaxN_6Den, 
## VmaxN_1DenFac, VmaxN_AOO, VmaxN_NOO, VmaxNH4_AOX, VmaxNO2_AOX 
artvalue = 1 # set at 1 if do not want to change anything
print("artifical value * DIN Vmax", artvalue)

den_penalty = 1
VmaxN = 50.8

# nitrate reduction to nitrite (NO3 --> NO2), denstep = 1
f_n1Den = calc_y(DG1, DGom, ep, DGs)       # fraction of electrons used for biomass synthesis
y_n1Den = f_n1Den * dO/dB
y_n1NO3 = (f_n1Den/dB) / ((1.0-f_n1Den)/2.0) # yield of biomass per unit nitrate consumed (full equation for NO3->NO2 functional type)
e_n1Den = 1.0 / y_n1NO3         # mols NO2 produced per mol biomass synthesised
VmaxN_1Den = VmaxN        # max mol nitrate consum. rate / mol cell N / day at 20C. (Zakem et al., 2018)

# nitrite reduction to N2 (NO2 --> N2), denstep = 3 (2)
f_n2Den = calc_y(DG234, DGom, ep, DGs) * (1 - P)      # fraction of electrons used for biomass synthesis
y_n2Den = f_n2Den * dO/dB #* (1 - P)
y_n2NO2 = (f_n2Den/dB) / ((1.0-f_n2Den)/3.0)       # yield of biomass per unit nitrite consumed
e_n2Den = 1.0 / y_n2NO2         # moles N-N2 produced per mole biomass synthesised
VmaxN_2Den = VmaxN         # max mole Nitrite consum rate / mol cell N / day at 20C

# Full denitrification (NO3 --> N2), denstep = 4 (3)
f_n3Den = calc_y(DG1234, DGom, ep, DGs) * (1 - P * 2)      # fraction of electrons used for biomass synthesis
y_n3Den = f_n3Den * dO/dB #* (1 - P * 2)
y_n3NO3 = (f_n3Den/dB) / ((1.0-f_n3Den)/5.0) # yield of biomass per unit nitrate consumed (Zakem et al., 2019 A14)
e_n3Den = 1.0 / y_n3NO3         # moles N-N2 produced per mole biomass synthesised
VmaxN_3Den = VmaxN         # max mole nitrate consum. rate / mol cell N / day at 20C

# NEW nitrite reduction to N2O (NO2 --> N2O), denstep = 2 (1)
f_n4Den = calc_y(DG23, DGom, ep, DGs)     # fraction of electrons used for biomass synthesis
y_n4Den = f_n4Den * dO/dB
y_n4NO2 = (f_n4Den/dB) / ((1.0-f_n4Den)/2.0) # yield of biomass per unit nitrite consumed
e_n4Den = 1.0 / y_n4NO2         # moles N-N2O produced per mole biomass synthesised
VmaxN_4Den = VmaxN         # max mole nitrite consum. rate / mol cell N / day at 20C

# NEW N2O reduction to N2 (N2O --> N2), denstep = 1
f_n5Den = calc_y(DG4, DGom, ep, DGs)     # fraction of electrons used for biomass synthesis
y_n5Den = f_n5Den * dO/dB
y_n5N2O = (f_n5Den/dB) / ((1.0-f_n5Den)/1.0) # yield of biomass per unit N-N2O consumed
e_n5Den = 1.0 / y_n5N2O         # moles N-N2 produced per mole biomass synthesised
# Below uses max growth rate to define max N2O uptake rate, the other option is to use N2O diffusion to define max N2O uptake rate.
VmaxN_5Den = VmaxN         # max mole N-N2O consum. rate / mol cell N / day at 20C

# NEW nitrate reduction to N2O (NO3 --> N2O), denstep = 3 (2)
f_n6Den = calc_y(DG123, DGom, ep, DGs) * (1 - P)     # fraction of electrons used for biomass synthesis
y_n6Den = f_n6Den * dO/dB #* (1 - P)
y_n6NO3 = (f_n6Den/dB) / ((1.0-f_n6Den)/4.0) # yield of biomass per unit nitrate consumed
e_n6Den = 1.0 / y_n6NO3         # moles N-N2O produced per mole biomass synthesised
VmaxN_6Den = VmaxN 


# Bookend den (NO3 --> NO2, N2O-->N2), denstep = 2
f_n7Den_NO3 = calc_y(DG1, DGom, ep, DGs) * (1 - P)       # fraction of electrons used for biomass synthesis
y_n7Den_NO3 = f_n7Den_NO3 * dO/dB
y_n7NO3 = (f_n7Den_NO3/dB) / ((1.0-f_n7Den_NO3)/2.0) # yield of biomass per unit nitrate consumed (full equation for NO3->NO2 functional type)
e_n7Den_NO3 = 1.0 / y_n7NO3         # mols NO2 produced per mol biomass synthesised

f_n7Den_N2O = calc_y(DG4, DGom, ep, DGs) * (1 - P)     # fraction of electrons used for biomass synthesis
y_n7Den_N2O = f_n7Den_N2O * dO/dB
y_n7N2O = (f_n7Den_N2O/dB) / ((1.0-f_n7Den_N2O)/1.0) # yield of biomass per unit N-N2O consumed
e_n7Den_N2O = 1.0 / y_n7N2O         # moles N-N2 produced per mole biomass synthesised

VmaxN_7Den = VmaxN        # max mol nitrate consum. rate / mol cell N / day at 20C. (Zakem et al., 2018)


# Facultative heterotrophy (oxygen and nitrate reduction to nitrite)
fac_penalty = 0.8
y_oHetFac = y_oHet * fac_penalty
f_oHetFac = y_oHetFac * dB/dO         # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2Fac = (f_oHetFac/dB) / ((1.0-f_oHetFac)/4.0)  # yield of biomass per unit oxygen reduced

y_n1DenFac = y_n1Den * fac_penalty
f_n1DenFac = y_n1DenFac * dB/dO
y_n1NO3Fac = (f_n1DenFac/dB) / ((1.0-f_n1DenFac)/2.0)
VmaxN_1DenFac = VmaxN        # max uptake rate of mol NO3 / mol cell N / day at 20C

# Chemoautotrophic ammonia oxidation (NH3 --> NO2)
y_nAOO = 0.0245         # mol N biomass per mol NH4 (Bayer et al. 2022; Zakem et al. 2022)
d_AOO = 4*5 + 7 - 2*2 -3*1  # number of electrons produced
#d_AOO = 4*4 + 7 - 2*2 -3*1  # number of electrons produced
f_AOO = y_nAOO / (6*(1/d_AOO - y_nAOO/d_AOO))         # fraction of electrons going to biomass synthesis from electron donor (NH4) (Zakem et al. 2022)
y_oAOO = f_AOO/d_AOO / ((1-f_AOO)/4.0)                # mol N biomass per mol O2 !!not O-O2 (Bayer et al. 2022; Zakem et al. 2022)
VmaxN_AOO = VmaxN #mumax_AOO / y_nAOO            # max uptake rate of mol NH4 / mol cell N / day

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOO = 0.0126         # mol N biomass per mol NO2 (Bayer et al. 2022)
d_NOO = 4*5 + 7 - 2*2 - 3*1
#d_NOO = 3.4*4 + 7 - 2*2 - 3*1
f_NOO = (y_nNOO * d_NOO) /2          # fraction of electrons going to biomass synthesis from electron donor (NO2) (Zakem et al. 2022)
y_oNOO = 4*f_NOO*(1-f_NOO)/d_NOO         # mol N biomass per mol O2 (Bayer et al. 2022)
VmaxN_NOO = VmaxN # mumax_NOO / y_nNOO = 1/0.0126 = 79.4

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOOan = 0.0126 * 0.7 #* 61.1/74.1       
# f_NOOan = (y_nNOOan * d_NOO) /2          # fraction of electrons going to biomass synthesis from electron donor (NO2)
# y_anNOO = 4*f_NOOan*(1-f_NOOan)/d_NOO

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_nh4AOX = 1./75                  # mol N biomass per mol NH4 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
y_no2AOX = 1./89                  # mol N biomass per mol NO2 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
e_n2AOX = 150                     # mol N-N2 formed per mol biomass N synthesised ***Rounded to nearest whole number
e_no3AOX = 13                     # mol NO3 formed per mol biomass N synthesised ***Rounded to nearest whole number

VmaxNH4_AOX = VmaxN #mumax_AOX / y_nh4AOX              # max uptake rate of mol NH4 / mol cell N / day
VmaxNO2_AOX = VmaxN #mumax_AOX / y_no2AOX              # max uptake rate of mol NO2 / mol cell N / day


### 4) Half-saturation constants (since the input conc will be in µM = mmol/m3, k will also be in µM)
K_s = 0.1           # OM (organic nitrogen (uncertain) uM)
# Copiotroph
K_sC = 2# #Default use 2 instead of 1 to make sure all OM* of copio > that of oligo (based on Zakem at al. 2020 Nat comm: K of copi = 10*oligo
# end

VmaxN_DenC = VmaxN * 2 #copiotrophic denitrifiers N included in the model file following Zakem et al. 2020

# use the same K for all DIN
K_n_universal = 0.1
if K_n_all == 1:    
    K_n_Den = K_n_universal #4.0 #K_n_universal #4.0       # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995)
    K_n_AOO = K_n_universal #0.1 #K_n_universal #0.1       # Martens-Habbena et al. 2009 Nature
    K_n_NOO = K_n_universal #0.1 #K_n_universal #0.1       # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
    K_nh4_AOX = K_n_universal #0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua
    K_no2_AOX = K_n_universal #0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua actually finds a K_no2 of 3.0 uM, but this excludes anammox completely in our experiments
else:
    K_n_Den = 4#1#4.0 #K_n_universal #4.0       # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995)
    K_n_DenC = K_n_Den * 10 #copiotrophic denitrifiers N-included in the model file following Zakem et al. 2020
    K_n_AOO = 0.1 #K_n_universal #0.1       # Martens-Habbena et al. 2009 Nature
    K_n_NOO = 0.1#0.1 #K_n_universal #0.1       # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
    K_nh4_AOX = 0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua
    K_no2_AOX = 0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua actually finds a K_no2 of 3.0 uM, but this excludes anammox completely in our experiments
  

# set K = 0 for N2O or O2 if want to model their uptake rate as a linear function (only constrained by diffusion)
K_n2o_Den = 0.3*2 #1e-4 # 0.3*2   # *2 convert µM N2O into µM N-N2O based on (Sun et al., 2021) in ODZ k = 0.3 µM N2O, in oxic layer = 1.4~2.8 µM
K_n2o_DenC = K_n2o_Den*10#copiotrophic denitrifiers N-included in the model file

K_o2_aer = 0.2   # (µM-O2) 10 to 200 nM at extremely low oxygen concentrations (Tiano et al., 2014); 200 nM (low affinity terminal oxidases) (Morris et al., 2013)
K_o2_aoo = 0.333  # (µM-O2) 333 nM at oxic-anoxic interface (Bristow et al., 2016)
K_o2_noo = 0.8   # (µM-O2) 778 nM at oxic-anoxic interface or 0.5 nM and 1750 nM breaking into two M-M curves (Bristow et al., 2016)



# ### Check useages of N in reactions
# print("")
# print("moles OrgN used in aerobic heterotrophy =",1/y_oHet)
# print("moles OrgN used in facultative aerobic heterotrophy =",1/y_oHetFac)
# print("moles OrgN used in facultative nitrate reducer (NO3 --> NO2) =",1/y_n1DenFac)

# print("moles OrgN used in denitrification1 (NO3 --> NO2) =",1/y_n1Den)
# print("moles OrgN used in denitrification2 (NO2 --> N2) =",1/y_n2Den)
# print("moles OrgN used in denitrification3 (NO3 --> N2) =",1/y_n3Den)
# print("moles OrgN used in denitrification4 (NO2 --> N2O) =",1/y_n4Den)
# print("moles OrgN used in denitrification5 (N2O --> N2) =",1/y_n5Den)
# print("moles OrgN used in denitrification6 (NO3 --> N2O) =",1/y_n6Den)

# print("moles NO3 used in denitrification1 (NO3 --> NO2) =",1/y_n1NO3)
# print("moles NO2 used in denitrification2 (NO2 --> N2) =",1/y_n2NO2)
# print("moles NO3 used in denitrification3 (NO3 --> N2) =",1/y_n3NO3)
# print("moles NO2 used in denitrification4 (NO2 --> N2O) =",1/y_n4NO2)
# # the below might be changed if use diffusion to define the max N2O uptake
# print("moles NO2 used in denitrification5 (N2O --> N2) =",1/y_n5N2O)
# print("moles NO3 used in denitrification6 (NO3 --> N2O) =",1/y_n6NO3)

# print("moles NH4 used in ammonia oxidation =",1/y_nAOO)
# print("moles NO2 used in nitrite oxidation =",1/y_nNOO)
# print("moles NH4+NO2 used in anammox =",(1/y_nh4AOX + 1/y_no2AOX), "and produced as Biomass, NO3 and N2 in anammox =",(e_no3AOX + e_n2AOX + 1))

# ### Check usages of oxygen in reactions
# print("")
# print("moles O2 used in aerobic heterotrophy =",1/y_oO2)
# print("moles O2 used in facultative aerobic heterotrophy =",1/y_oO2Fac)
# print("moles O2 used in ammonia oxidation =",1/y_oAOO)
# print("moles O2 used in nitrite oxidation =",1/y_oNOO)

print("y_oO2Het =", y_oO2)
print("y_oAOO =", y_oAOO)
print("y_oNOO =", y_oNOO)

print("y_nAOO =", y_nAOO)
print("y_nNOO =", y_nNOO)
print("y_nh4AOX =", y_nh4AOX)
print("y_no2AOX =", y_no2AOX)

print("OM yield for heterotrophs")  
print("y_oHet =", y_oHet)
#print("y_oHetFac =", y_oHetFac)
#print("y_n1DenFac =", y_n1DenFac)
print("y_n1Den =", y_n1Den)
print("y_n2Den =", y_n2Den)
print("y_n3Den =", y_n3Den)
print("y_n4Den =", y_n4Den)
print("y_n5Den =", y_n5Den)
print("y_n6Den =", y_n6Den)
print("y_heterAverage =", (y_oHet+y_n1Den+y_n2Den+y_n3Den+y_n4Den+y_n5Den+y_n6Den)/7)

print("DIN yield for dens")  
print("y_n1Den =", y_n1NO3)
print("y_n2Den =", y_n2NO2)
print("y_n3Den =", y_n3NO3)
print("y_n4Den =", y_n4NO2)
print("y_n5Den =", y_n5N2O)
print("y_n6Den =", y_n6NO3)

#O2
VmaxO2_aer = mumax_Het / y_oO2
VmaxO2_AOO = mumax_AOO / y_oAOO 
VmaxO2_NOO = mumax_NOO / y_oNOO
print(VmaxO2_aer)
print(VmaxO2_AOO) 
print(VmaxO2_NOO)
print(mumax_Het)
print(mumax_AOO)
print(mumax_NOO)

#OM
print(VmaxS)
print(VmaxS)
print(VmaxS)
print(VmaxS)
print(VmaxS)
print(VmaxS)
print(VmaxS)
print(VmaxS * y_oHet)
print(VmaxS * y_n1Den)
print(VmaxS * y_n2Den)
print(VmaxS * y_n3Den)
print(VmaxS * y_n4Den)
print(VmaxS * y_n5Den)
print(VmaxS * y_n6Den)
# DIN
print(VmaxNH4_AOX)
print(VmaxNO2_AOX)
print(VmaxNH4_AOX * y_nh4AOX)
print(VmaxNO2_AOX * y_no2AOX)

print(VmaxN_AOO)
print(VmaxN_AOO * y_nAOO)

print(VmaxN_NOO)
print(VmaxN_NOO * y_nNOO)

print(VmaxN_1Den)
print(VmaxN_1Den * y_n1NO3)
print(VmaxN_2Den)
print(VmaxN_2Den* y_n2NO2)
print(VmaxN_3Den)
print(VmaxN_3Den* y_n3NO3)
print(VmaxN_4Den)
print(VmaxN_4Den * y_n4NO2)
print(VmaxN_6Den)
print(VmaxN_6Den* y_n6NO3)
#N2O
print(VmaxN_5Den)
print(VmaxN_5Den * y_n5N2O)

## Caculate K for NO2 denitrifiers to make it has a larger NO2* than anammox or than NOB
# Loss_dil = 0.04

# K_n_Den4_AOX = K_no2_AOX * (VmaxN_4Den * y_n4NO2 - Loss_dil)/(VmaxNO2_AOX * y_no2AOX - Loss_dil)
# K_n_Den4_NOO = K_n_NOO * (VmaxN_4Den * y_n4NO2 - Loss_dil)/(VmaxN_NOO * y_nNOO - Loss_dil)
# print("K_n_Den4_AOX =", K_n_Den4_AOX)
# print("K_n_Den4_NOO =", K_n_Den4_NOO)
# K_n_Den4_AOX = 0.1459387619142415
# K_n_Den4_NOO = 0.12908666696179774

# f_n3Den = calc_y(DG1234, DGom, ep, DGs) * (1 - P * 2)   
# f_n6Den = calc_y(DG123, DGom, ep, DGs) * (1 - P) 
   
# diff = 5 * calc_y(DG1234, DGom, ep, DGs) * (1 - P * 2) / (1.0-calc_y(DG1234, DGom, ep, DGs) * (1 - P * 2)) - 4 * calc_y(DG123, DGom, ep, DGs) * (1 - P)  / (1.0-calc_y(DG123, DGom, ep, DGs) * (1 - P) ) #y_n3NO3 = y_n6NO3
    
# print("diff", diff)

