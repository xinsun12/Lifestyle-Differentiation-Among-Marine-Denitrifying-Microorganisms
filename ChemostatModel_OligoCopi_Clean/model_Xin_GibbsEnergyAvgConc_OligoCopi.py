# -*- coding: utf-8 -*-
"""
Created in July 2022

Purpose
-------
    A 0D chemostat model of redox reactions occuring in marine OMZs
    
    This model predicts the outcome of microbial populations competing for resources
    in OMZs using the energetics and stoichiometries of the reactions they
    perform. 
    We are particularly interested in the dynamics of N2O in OMZs.

@author: Emily Zakem, Pearse Buchanan and Xin Sun
"""

### imports
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
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
             N2Oammonia):
    
    # if Copiotrophs also in terms of N (if not, use the other value)
    VmaxN_DenC = VmaxN_1Den * 2 #50.8
    K_n_DenC = K_n_Den * 10 #K_n_DenC = K_n_Den
    K_n2o_DenC = K_n2o_Den * 10 #K_n2o_DenC = K_n2o_Den
    VmaxO2_aerC = mumax_Het / y_oO2 * 4 # VmaxO2_aerC = mumax_Het / y_oO2
    K_o2_aerC = K_o2_aer * 20 #K_o2_aerC = K_o2_aer
    
    # transfer initial inputs to model variables
    m_Sd = initialOM#in_Sd #10 change intial OM m_Sd to 10 µM and see what happens 
    m_Sp = in_Sp
    m_O2 = in_O2 # make sure the unit is O2 not O for m_O2
    m_NO3 = in_NO3
    m_NO2 = initialNO2
    m_NH4 = in_NH4
    m_N2 = in_N2  # make sure the unit is N-N2 for m_N2
    m_N2O = in_N2O  # make sure the unit is N-N2O (mmol-N/m3 = µM) for m_N2O 
    m_bHet = in_bHet
    m_bFac = in_bFac
    m_b1Den = in_b1Den
    m_b2Den = in_b2Den
    m_b3Den = in_b3Den
    m_b4Den = in_b4Den
    m_b5Den = in_b5Den
    m_b6Den = in_b6Den
    m_b7Den = in_b7Den
    #Copio
    m_bHetC = in_bHetC
    m_b1DenC = in_b1DenC
    m_b2DenC = in_b2DenC
    m_b3DenC = in_b3DenC
    m_b4DenC = in_b4DenC # newly added
    m_b5DenC = in_b5DenC # newly added
    m_b6DenC = in_b6DenC # newly added
    #end
    m_bAOO = in_bAOO
    m_bNOO = in_bNOO
    m_bNOOan = in_bNOOan
    m_bAOX = in_bAOX
    m_an = in_an
    
    # set the output arrays 
    out_Sd = np.ones((int(nn_output)+1)) * np.nan
    out_Sp = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NO3 = np.ones((int(nn_output)+1)) * np.nan
    out_NO2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_N2 = np.ones((int(nn_output)+1)) * np.nan
    out_N2O = np.ones((int(nn_output)+1)) * np.nan
    out_bHet = np.ones((int(nn_output)+1)) * np.nan
    out_bFac = np.ones((int(nn_output)+1)) * np.nan
    out_b1Den = np.ones((int(nn_output)+1)) * np.nan
    out_b2Den = np.ones((int(nn_output)+1)) * np.nan
    out_b3Den = np.ones((int(nn_output)+1)) * np.nan
    out_b4Den = np.ones((int(nn_output)+1)) * np.nan
    out_b5Den = np.ones((int(nn_output)+1)) * np.nan
    out_b6Den = np.ones((int(nn_output)+1)) * np.nan
    out_b7Den = np.ones((int(nn_output)+1)) * np.nan
    #add copi (1 aerHet + 6 dens)
    out_bHetC = np.ones((int(nn_output)+1)) * np.nan
    out_b1DenC = np.ones((int(nn_output)+1)) * np.nan
    out_b2DenC = np.ones((int(nn_output)+1)) * np.nan
    out_b3DenC = np.ones((int(nn_output)+1)) * np.nan
    out_b4DenC = np.ones((int(nn_output)+1)) * np.nan
    out_b5DenC = np.ones((int(nn_output)+1)) * np.nan
    out_b6DenC = np.ones((int(nn_output)+1)) * np.nan
    #end
    out_bAOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOOan = np.ones((int(nn_output)+1)) * np.nan
    out_bAOX = np.ones((int(nn_output)+1)) * np.nan
    out_uHet = np.ones((int(nn_output)+1)) * np.nan
    out_uFac = np.ones((int(nn_output)+1)) * np.nan
    out_u1Den = np.ones((int(nn_output)+1)) * np.nan
    out_u2Den = np.ones((int(nn_output)+1)) * np.nan
    out_u3Den = np.ones((int(nn_output)+1)) * np.nan
    out_u4Den = np.ones((int(nn_output)+1)) * np.nan
    out_u5Den = np.ones((int(nn_output)+1)) * np.nan
    out_u6Den = np.ones((int(nn_output)+1)) * np.nan
    out_u7Den = np.ones((int(nn_output)+1)) * np.nan
    #add copi (1 aerHet + 6 dens)
    out_uHetC = np.ones((int(nn_output)+1)) * np.nan
    out_u1DenC = np.ones((int(nn_output)+1)) * np.nan
    out_u2DenC = np.ones((int(nn_output)+1)) * np.nan
    out_u3DenC = np.ones((int(nn_output)+1)) * np.nan
    out_u4DenC = np.ones((int(nn_output)+1)) * np.nan
    out_u5DenC = np.ones((int(nn_output)+1)) * np.nan
    out_u6DenC = np.ones((int(nn_output)+1)) * np.nan       
    #end
    out_uAOO = np.ones((int(nn_output)+1)) * np.nan
    out_uNOO = np.ones((int(nn_output)+1)) * np.nan
    out_uNOOan = np.ones((int(nn_output)+1)) * np.nan
    out_uAOX = np.ones((int(nn_output)+1)) * np.nan
    out_facaer = np.ones((int(nn_output)+1)) * np.nan
    out_rHet = np.ones((int(nn_output)+1)) * np.nan
    out_rHetAer = np.ones((int(nn_output)+1)) * np.nan
    out_rO2C = np.ones((int(nn_output)+1)) * np.nan
    out_r1Den = np.ones((int(nn_output)+1)) * np.nan
    out_r2Den = np.ones((int(nn_output)+1)) * np.nan
    out_r3Den = np.ones((int(nn_output)+1)) * np.nan
    out_r4Den = np.ones((int(nn_output)+1)) * np.nan
    out_r5Den = np.ones((int(nn_output)+1)) * np.nan
    out_r6Den = np.ones((int(nn_output)+1)) * np.nan
    #out_r7Den = np.ones((int(nn_output)+1)) * np.nan
    #add copi (6 dens) 
    out_r1DenC = np.ones((int(nn_output)+1)) * np.nan
    out_r2DenC = np.ones((int(nn_output)+1)) * np.nan
    out_r3DenC = np.ones((int(nn_output)+1)) * np.nan
    out_r4DenC = np.ones((int(nn_output)+1)) * np.nan
    out_r5DenC = np.ones((int(nn_output)+1)) * np.nan
    out_r6DenC = np.ones((int(nn_output)+1)) * np.nan      
    #end
    out_rAOO = np.ones((int(nn_output)+1)) * np.nan
    out_rNOO = np.ones((int(nn_output)+1)) * np.nan
    out_rNOOan = np.ones((int(nn_output)+1)) * np.nan
    out_rAOX = np.ones((int(nn_output)+1)) * np.nan
    
    out_rN2Oammonia = np.ones((int(nn_output)+1)) * np.nan
    
    
    # set the array for recording average activity of facultative anaerobes
    interval = int((1/dt * out_at_day))
    facaer = np.ones((interval)) * np.nan
    
    # record the initial conditions
    i = 0
    out_Sd[i] = m_Sd 
    out_Sp[i] = m_Sp
    out_O2[i] = m_O2
    out_NO3[i] = m_NO3
    out_NO2[i] = m_NO2 
    out_NH4[i] = m_NH4 
    out_N2[i] = m_N2
    out_N2O[i] = m_N2O
    out_bHet[i] = m_bHet
    out_bFac[i] = m_bFac
    out_b1Den[i] = m_b1Den
    out_b2Den[i] = m_b2Den
    out_b3Den[i] = m_b3Den 
    out_b4Den[i] = m_b4Den
    out_b5Den[i] = m_b5Den
    out_b6Den[i] = m_b6Den
    out_b7Den[i] = m_b7Den
    # copi (1 aerHet + 6 dens)
    out_bHetC[i] = m_bHetC
    out_b1DenC[i] = m_b1DenC
    out_b2DenC[i] = m_b2DenC
    out_b3DenC[i] = m_b3DenC 
    out_b4DenC[i] = m_b4DenC # newly added
    out_b5DenC[i] = m_b5DenC # newly added
    out_b6DenC[i] = m_b6DenC # newly added
    # end
    out_bAOO[i] = m_bAOO
    out_bNOO[i] = m_bNOO
    out_bNOOan[i] = m_bNOOan
    out_bAOX[i] = m_bAOX
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        ## Different equations for uptake rates (p) in the unit of mol XX / day / mol-BiomassN
        if Gasuptake == 1:
            # 1. Tend not to use this ‘MMdiff’: Define Vmax (maximum uptake rates) by diffusion, then V is described by M-M equation:
            p_O2_aer = (po_aer * m_O2) * m_O2 / (K_o2_aer + m_O2)     # O2 uptake rate for aerobic heterotrophs and facultative?
            p_O2_aoo = (po_aoo * m_O2) * m_O2 / (K_o2_aoo + m_O2)     # O2 uptake rate for AOA
            p_O2_noo = (po_noo * m_O2) * m_O2 / (K_o2_noo + m_O2)     # O2 uptake rate for NOB
            p_N2O_den = (pn2o_den * m_N2O) * m_N2O / (K_n2o_Den + m_N2O)       # mol N-N2O / day / mol-BiomassN (m_N2O is in the unit of N-N2O (mmol-N/m3 = µM))        
        else:
            if Gasuptake == 2:
                # 2. ‘diff’: Plot out V only defined by diffusion, or turn k = 0 for equations in 1 get equations in 2 
                p_O2_aer = (po_aer * m_O2)      # O2 uptake rate for aerobic heterotrophs and facultative?
                p_O2_aoo = (po_aoo * m_O2)      # O2 uptake rate for AOA
                p_O2_noo = (po_noo * m_O2)      # O2 uptake rate for NOB
                p_N2O_den = (pn2o_den * m_N2O)       # mol N-N2O / day / mol-BiomassN (m_N2O is in the unit of N-N2O (mmol-N/m3 = µM)) 
            else:
                if Gasuptake == 3:
                    # 3.	‘MMµ’: Define Vmax by µmax and yield, like the other Vmax in the model, then add M-M equation to that.
                    VmaxO2_aer = mumax_Het / y_oO2
                    p_O2_aer = VmaxO2_aer * m_O2 / (K_o2_aer + m_O2)
                    p_O2_aerC = VmaxO2_aerC / y_oO2 * m_O2 / (K_o2_aerC + m_O2)
            
                    VmaxO2_AOO = mumax_AOO / y_oAOO  
                    p_O2_aoo = VmaxO2_AOO * m_O2 / (K_o2_aoo + m_O2)
            
                    VmaxO2_NOO = mumax_NOO / y_oNOO
                    p_O2_noo = VmaxO2_NOO * m_O2 / (K_o2_noo + m_O2)
            
                    #VmaxN_5Den = mumax_Het * den_penalty / y_n5N2O        # max mole N-N2O consum / mol cell N / day
                    p_N2O_den = VmaxN_5Den * m_N2O / (K_n2o_Den + m_N2O)
                    p_N2O_denC = VmaxN_DenC * m_N2O / (K_n2o_DenC + m_N2O)

      
        #p_Sp = VmaxS * m_Sp / (K_s + m_Sp)  # did not include particular OM # mol Org / day (Vmax: max mol XX consum per mol-BioN per day)
        p_Sd = VmaxS * m_Sd / (K_s + m_Sd)                               # mol Org / day
        p_SdC = VmaxSC * m_Sd / (K_sC + m_Sd)              #Copiotroph
        p_FacNO3 = VmaxN_1DenFac * m_NO3 / (K_n_Den + m_NO3)             # mol NO3 / day
        p_1DenNO3 = VmaxN_1Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_2DenNO2 = VmaxN_2Den * m_NO2 / (K_n_Den2 + m_NO2)               # mol NO2 / day
        p_3DenNO3 = VmaxN_3Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_4DenNO2 = VmaxN_4Den * m_NO2 / (K_n_Den4 + m_NO2)               # mol NO2 / day
        p_6DenNO3 = VmaxN_6Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_NH4_AOO = VmaxN_AOO * m_NH4 / (K_n_AOO + m_NH4)   # mol NH4 / day
        p_NO2_NOO = VmaxN_NOO * m_NO2 / (K_n_NOO + m_NO2)   # mol NO2 / day
        p_NH4_AOX = VmaxNH4_AOX * m_NH4 / (K_nh4_AOX + m_NH4)     # mol NH4 / day
        p_NO2_AOX = VmaxNO2_AOX * m_NO2 / (K_no2_AOX + m_NO2)     # mol NO2 / day
        

        p_1DenNO3C = VmaxN_DenC * m_NO3 / (K_n_DenC + m_NO3)               # mol NO3 / day
        p_2DenNO2C = VmaxN_DenC * m_NO2 / (K_n_DenC + m_NO2)               # mol NO2 / day
        p_3DenNO3C = VmaxN_DenC * m_NO3 / (K_n_DenC + m_NO3)               # mol NO3 / day
        p_4DenNO2C = VmaxN_DenC * m_NO2 / (K_n_DenC + m_NO2)               # mol NO2 / day
        p_6DenNO3C = VmaxN_DenC * m_NO3 / (K_n_DenC + m_NO3)               # mol NO3 / day
        
        if GrowthRateMM == 1:
            # growth rates (u) in the unit of d^(-1), determined by the min rate of the substrate (e.g., OM, DIN, O2) uptake
            u_Het = np.fmax(0.0, np.fmin(p_Sd * y_oHet, p_O2_aer * y_oO2))          # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
            u_FacO2 = np.fmax(0.0, np.fmin(p_Sd * y_oHetFac, p_O2_aer * y_oO2Fac))  # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
            u_FacNO3 = np.fmax(0.0, np.fmin(p_Sd * y_n1DenFac, p_FacNO3 * y_n1NO3Fac)) # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
            u_1Den = np.fmax(0.0, np.fmin(p_Sd * y_n1Den, p_1DenNO3 * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
            u_2Den = np.fmax(0.0, np.fmin(p_Sd * y_n2Den, p_2DenNO2 * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
            u_3Den = np.fmax(0.0, np.fmin(p_Sd * y_n3Den, p_3DenNO3 * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3       
            u_4Den = np.fmax(0.0, np.fmin(p_Sd * y_n4Den, p_4DenNO2 * y_n4NO2))         # (NO2-->N2O)
            u_5Den = np.fmax(0.0, np.fmin(p_Sd * y_n5Den, p_N2O_den * y_n5N2O))         # (N2O-->N2)
            u_6Den = np.fmax(0.0, np.fmin(p_Sd * y_n6Den, p_6DenNO3 * y_n6NO3))         # (NO2-->N2O)
            u_7Den = np.fmax(np.fmax(0.0, np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3)), np.fmax(0.0, np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O)))         # (bookend: NO3-->NO2, N2O-->N2)
            
            #Copiotrophs (1 aerHet + 6 dens)
            u_HetC = np.fmax(0.0, np.fmin(p_SdC * y_oHet, p_O2_aer * y_oO2))          # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
            #u_FacO2 = np.fmax(0.0, np.fmin(p_Sd * y_oHetFac, p_O2_aer * y_oO2Fac))  # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
            #u_FacNO3 = np.fmax(0.0, np.fmin(p_Sd * y_n1DenFac, p_FacNO3 * y_n1NO3Fac)) # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
            
            # u_1DenC = np.fmax(0.0, np.fmin(p_SdC * y_n1Den, p_1DenNO3 * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
            # u_2DenC = np.fmax(0.0, np.fmin(p_SdC * y_n2Den, p_2DenNO2 * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
            # u_3DenC = np.fmax(0.0, np.fmin(p_SdC * y_n3Den, p_3DenNO3 * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3       
            # u_4DenC = np.fmax(0.0, np.fmin(p_SdC * y_n4Den, p_4DenNO2 * y_n4NO2))         # 
            # u_5DenC = np.fmax(0.0, np.fmin(p_SdC * y_n5Den, p_N2O_den * y_n5N2O))         # 
            # u_6DenC = np.fmax(0.0, np.fmin(p_SdC * y_n6Den, p_6DenNO3 * y_n6NO3))  
            ##OR
            u_1DenC = np.fmax(0.0, np.fmin(p_SdC * y_n1Den, p_1DenNO3C * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
            u_2DenC = np.fmax(0.0, np.fmin(p_SdC * y_n2Den, p_2DenNO2C * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
            u_3DenC = np.fmax(0.0, np.fmin(p_SdC * y_n3Den, p_3DenNO3C * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3       
            u_4DenC = np.fmax(0.0, np.fmin(p_SdC * y_n4Den, p_4DenNO2C * y_n4NO2))         # 
            u_5DenC = np.fmax(0.0, np.fmin(p_SdC * y_n5Den, p_N2O_denC * y_n5N2O))         # 
            u_6DenC = np.fmax(0.0, np.fmin(p_SdC * y_n6Den, p_6DenNO3C * y_n6NO3)) 
            #end
            u_AOO = np.fmax(0.0, np.fmin(p_NH4_AOO * y_nAOO, p_O2_aoo * y_oAOO))    # mol NH4 / day * mol Biomass / mol NH4 || mol O2 / day * mol Biomass / mol O2
            u_NOO = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOO, p_O2_noo * y_oNOO))    # mol NO2 / day * mol Biomass / mol NO2 || mol O2 / day * mol Biomass / mol O2
            #u_NOOan = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOOan, p_NO2_NOO * y_nNOOan)) # dismutation
            u_NOOan = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOOan, mumax_NOO * m_an / (0.001 + m_an))) #p_Sd * (y_nNOOan *0.0075/16 * 1/2)
            #p_O2_noo = mumax_NOO * m_O2 / (K_o2_noo + m_O2)
            
            u_AOX = np.fmax(0.0, np.fmin(p_NO2_AOX * y_no2AOX, p_NH4_AOX * y_nh4AOX)) # mol NO2 / day * mol Biomass / mol NO2 || mol NH4 / day * mol Biomass / mol NH4
        #else:
            #if GrowthRateMM == 2: #max growth rate of denitrifiers are unknown--> max (VmaxS * y_n1Den, Vmax_N * y_n1NO3)?
                # growth rates (u) in the unit of d^(-1), determined by all substrates
                #u_Het = mumax_Het * m_Sd / (K_s + m_Sd) * m_O2 / (K_o2_aer + m_O2)
                #u_FacO2 = mumax_Het * fac_penalty * m_Sd / (K_s + m_Sd) * m_O2 / (K_o2_aer + m_O2)    
                #u_FacNO3 = mumax_Het * fac_penalty * m_Sd / (K_s + m_Sd) * m_NO3 / (K_n_Den + m_NO3)        
                #u_1Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_NO3 / (K_n_Den + m_NO3)
                #u_2Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_NO2 / (K_n_Den + m_NO2)
                #u_3Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_NO3 / (K_n_Den + m_NO3)
                #u_4Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_NO2 / (K_n_Den + m_NO2)
                #u_5Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_N2O / (K_n2o_Den + m_N2O) 
                #u_6Den = mumax_Het * den_penalty * m_Sd / (K_s + m_Sd) * m_NO3 / (K_n_Den + m_NO3)        
                #u_AOO = mumax_AOO * m_O2 / (K_o2_aoo + m_O2) * m_NH4 / (K_n_AOO + m_NH4)
                #u_NOO = mumax_NOO * m_O2 / (K_o2_noo + m_O2) * m_NO2 / (K_n_NOO + m_NO2)
                #u_AOX = mumax_AOX * m_NH4 / (K_nh4_AOX + m_NH4) * m_NO2 / (K_no2_AOX + m_NO2)
        
        
        ### BIOMASS TYPES & (need to uptake below stoichiometry) THEIR STOICHIOMETRIES (already encoded within yields, which are inputs)
        # 1. Aerobic heterotrophy ( 7.1-OM + 47-O2 --> B + 42-CO2 + 6.1-NH4)
        # 2. Facultative anaerobes ( 8.9-OM + 60-O2 --> B + 53-CO2 + 7.9-NH4)
        # 3. Denitrification ( 7.9-OM + 105-NO3 --> B + 47-CO2 + 6.9-NH4 + 105-NO2)
        # 4. Denitrification ( 7.9-OM + 70-NO2 --> B + 47-CO2 + 6.9-NH4 + 35-N2)
        # 5. Denitrification ( 7.9-OM + 42-NO3 --> B + 47-CO2 + 6.9-NH4 + 21-N2)
        # 54.Denitrifcation (NO2-->N2O)
        # 55.Denitrification (N2O-->N2)
        # 56.Denitrification (NO3-->N2O)
        
        # 6. Ammonia oxidation ( 112-NH4 + 5-CO2 + 162-O2 --> B + 111-NO2)
        # 7. Nitrite oxidation ( 334-NO2 + 5-CO2 + 162-O2 --> B + 334-NO3)
        # 8. Anammox ( 154-NH4 + 216-NO2 + 5-CO2 --> B + 42-NO3 + 163-N2)
        
        ###
        ### Change in state variables per day (ddt)
        ###
        
        # deal with facultative bacteria first
        if u_FacO2 >= u_FacNO3:
            u_Fac = u_FacO2
            ddt_Sd_Fac = u_Fac * m_bFac / y_oHetFac
            ddt_O2_Fac = u_Fac * m_bFac / y_oO2Fac
            ddt_NO3_Fac = 0.0
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_oHetFac - 1)
            facaer[int(t % interval)] = 1.0
        else:
            u_Fac = u_FacNO3
            ddt_Sd_Fac = u_Fac * m_bFac / y_n1DenFac # OM uptake rate (µM-N) of facultative nitrate reducer
            ddt_O2_Fac = 0.0
            ddt_NO3_Fac = u_Fac * m_bFac / y_n1NO3Fac
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_n1DenFac - 1)
            facaer[int(t % interval)] = 0.0
            
        
        ### rates
        # aerobic OM uptake rate (add Copio 1aer)
        aer_heterotrophy = u_Het * m_bHet / y_oHet      \
                           + u_HetC * m_bHetC / y_oHet \
                           + ddt_Sd_Fac
        # total OM uptake rate, add new functional types (add Copio 1aer + 6 den)
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):      
            heterotrophy = u_Het * m_bHet / y_oHet      \
                           + ddt_Sd_Fac                 \
                           + u_1Den * m_b1Den / y_n1Den \
                           + u_2Den * m_b2Den / y_n2Den \
                           + u_3Den * m_b3Den / y_n3Den \
                           + u_4Den * m_b4Den / y_n4Den \
                           + u_5Den * m_b5Den / y_n5Den \
                           + u_6Den * m_b6Den / y_n6Den \
                           + u_7Den * m_b7Den / y_n7Den_NO3 \
                           + u_HetC * m_bHetC / y_oHet \
                           + u_1DenC * m_b1DenC / y_n1Den \
                           + u_2DenC * m_b2DenC / y_n2Den \
                           + u_3DenC * m_b3DenC / y_n3Den \
                           + u_4DenC * m_b4DenC / y_n4Den \
                           + u_5DenC * m_b5DenC / y_n5Den \
                           + u_6DenC * m_b6DenC / y_n6Den 
                           #+ u_NOOan * m_bNOOan / (y_nNOOan *0.0075/16 * 1/2)
        else:
            heterotrophy = u_Het * m_bHet / y_oHet      \
                           + ddt_Sd_Fac                 \
                           + u_1Den * m_b1Den / y_n1Den \
                           + u_2Den * m_b2Den / y_n2Den \
                           + u_3Den * m_b3Den / y_n3Den \
                           + u_4Den * m_b4Den / y_n4Den \
                           + u_5Den * m_b5Den / y_n5Den \
                           + u_6Den * m_b6Den / y_n6Den \
                           + u_7Den * m_b7Den / y_n7Den_N2O \
                           + u_HetC * m_bHetC / y_oHet \
                           + u_1DenC * m_b1DenC / y_n1Den \
                           + u_2DenC * m_b2DenC / y_n2Den \
                           + u_3DenC * m_b3DenC / y_n3Den \
                           + u_4DenC * m_b4DenC / y_n4Den \
                           + u_5DenC * m_b5DenC / y_n5Den \
                           + u_6DenC * m_b6DenC / y_n6Den 
                           #+ u_NOOan * m_bNOOan / (y_nNOOan *0.0075/16 * 1/2)
        # Oxygen consumption rate (add Copio 1aer)
        oxy_consumption = u_Het * m_bHet / y_oO2    \
                          + u_HetC * m_bHetC / y_oO2 \
                          + ddt_O2_Fac              \
                          + u_AOO * m_bAOO / y_oAOO \
                          + u_NOO * m_bNOO / y_oNOO
        # DIN uptake rates (add Copio 6 den)
        ammonia_ox = u_AOO * m_bAOO / y_nAOO   # ammonia uptake rate by AOA
        nitrite_ox = u_NOO * m_bNOO / y_nNOO   # nitrite uptake rate by NOB
        nitrite_ox_an = u_NOOan * m_bNOOan / y_nNOOan   # nitrite uptake rate by NOB
        anammox_nh4 = u_AOX * m_bAOX / y_nh4AOX  # ammonia uptake rate by anammox bacteria
        anammox_no2 = u_AOX * m_bAOX / y_no2AOX  # nitrite uptake rate by anammox bacteria
        anammox_no3 = u_AOX * m_bAOX * e_no3AOX  # nitrate production rate by anammox bacteria
        
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):
            den_nar = u_1Den * m_b1Den / y_n1NO3        \
                      + u_1DenC * m_b1DenC / y_n1NO3    \
                      + u_7Den * m_b7Den / y_n7NO3      \
                      + ddt_NO3_Fac                       # nitrate uptake rate by nitrate reducers that produce nitrite (anaerobe, facultative)
            den_nir = u_2Den * m_b2Den / y_n2NO2 \
                      + u_2DenC * m_b2DenC / y_n2NO2  # nitrite uptake rate by NO2-->N2 
            den_full = u_3Den * m_b3Den / y_n3NO3 \
                       + u_3DenC * m_b3DenC / y_n3NO3 # nitrate uptake rate by NO3-->N2
            den_NitritetoN2O = u_4Den * m_b4Den / y_n4NO2 \
                               + u_4DenC * m_b4DenC / y_n4NO2 # nitrite uptake rate by NO2-->N2O                       
            den_N2OtoN2 = u_5Den * m_b5Den / y_n5N2O \
                          + u_5DenC * m_b5DenC / y_n5N2O # N2O uptake rate by N2O-->N2
            den_NitratetoN2O = u_6Den * m_b6Den / y_n6NO3 \
                               + u_6DenC * m_b6DenC / y_n6NO3 # nitrate uptake rate by NO3-->N2O
        else:
            den_nar = u_1Den * m_b1Den / y_n1NO3        \
                      + u_1DenC * m_b1DenC / y_n1NO3    \
                      + ddt_NO3_Fac                       # nitrate uptake rate by two types of nitrate reducers (anaerobe, facultative)
            den_nir = u_2Den * m_b2Den / y_n2NO2 \
                      + u_2DenC * m_b2DenC / y_n2NO2  # nitrite uptake rate by NO2-->N2 
            den_full = u_3Den * m_b3Den / y_n3NO3 \
                       + u_3DenC * m_b3DenC / y_n3NO3 # nitrate uptake rate by NO3-->N2
            den_NitritetoN2O = u_4Den * m_b4Den / y_n4NO2 \
                               + u_4DenC * m_b4DenC / y_n4NO2 # nitrite uptake rate by NO2-->N2O                       
            den_N2OtoN2 = u_5Den * m_b5Den / y_n5N2O \
                          + u_7Den * m_b7Den / y_n7N2O \
                          + u_5DenC * m_b5DenC / y_n5N2O # N2O uptake rate by N2O-->N2
            den_NitratetoN2O = u_6Den * m_b6Den / y_n6NO3 \
                               + u_6DenC * m_b6DenC / y_n6NO3 # nitrate uptake rate by NO3-->N2O
        
        #### include N2O from ammonia in the model
        if m_O2 <= (1/6.022*1e-18): #0.0685:
            N2OammoniaRate = (0.2/(1/6.022*1e-18) + 0.08)/100* ammonia_ox #3/100 * ammonia_ox #N2Oammonia
        else:
            N2OammoniaRate = (0.2/(m_O2) + 0.08)/100 * ammonia_ox #N2Oammonia       
        
        
        ## Below are the change of OM, DIN.. over time. For example, d[OM]/dt, d[NO3]/dt
        # Dissolved organic matter (consumed by 1, 2, 3, 4, 5, 54, 55)
        ddt_Sd = dil * (in_Sd - m_Sd) - heterotrophy
                 
        # Particulate organic matter (consumed by 1, 2, 3, 4, 5, 54, 55)
        ddt_Sp = dil * (in_Sp - m_Sp)   # particle associated POM not currently included in model
        
        # Dissolved oxygen (consumed by 1, 2, 6, 7)
        ddt_O2 = dil * (in_O2 - m_O2) - oxy_consumption
                 
        # Nitrate (consumed by 2&3, 5, 56) (produced by 7, 8) 
        ddt_NO3 = dil * (in_NO3 - m_NO3)        \
                 - den_nar                      \
                 - den_full                     \
                 - den_NitratetoN2O             \
                 + nitrite_ox                   \
                 + nitrite_ox_an                \
                 + anammox_no3        
        
        # Nitrite (consumed by 4, 7, 8, 54) (produced by 2&3, 6)
        ddt_NO2 = dil * (in_NO2 - m_NO2)        \
                 - den_nir                      \
                 - den_NitritetoN2O             \
                 - nitrite_ox                   \
                 - nitrite_ox_an                   \
                 - anammox_no2                  \
                 + den_nar                      \
                 + u_AOO * m_bAOO * (1./y_nAOO - 1)   # because it uses one mol of NH4 for biomass synthesis         
        ddt_an = dil * (in_an - m_an) - u_NOOan * m_bNOOan / (y_nNOOan * 0.5)
        
        
        
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):
            # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5, 54, 55, 56) (Add copio 1 aer + 6 den)
            ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                     - ammonia_ox                   \
                     - anammox_nh4                  \
                     - u_NOO * m_bNOO               \
                     - u_NOOan * m_bNOOan               \
                     + u_Het * m_bHet * (1./y_oHet - 1)    \
                     + ddt_NH4_Fac                         \
                     + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                     + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                     + u_3Den * m_b3Den * (1./y_n3Den - 1) \
                     + u_4Den * m_b4Den * (1./y_n4Den - 1) \
                     + u_5Den * m_b5Den * (1./y_n5Den - 1) \
                     + u_6Den * m_b6Den * (1./y_n6Den - 1) \
                     + u_7Den * m_b7Den * (1./y_n7Den_NO3 - 1) \
                     + u_HetC * m_bHetC * (1./y_oHet - 1)    \
                     + u_1DenC * m_b1DenC * (1./y_n1Den - 1) \
                     + u_2DenC * m_b2DenC * (1./y_n2Den - 1) \
                     + u_3DenC * m_b3DenC * (1./y_n3Den - 1) \
                     + u_4DenC * m_b4DenC * (1./y_n4Den - 1) \
                     + u_5DenC * m_b5DenC * (1./y_n5Den - 1) \
                     + u_6DenC * m_b6DenC * (1./y_n6Den - 1) \
            # Dinitrogen gas (produced by 4, 5, 8, 55) (Add copio 3 den)
            ddt_N2 = dil * (in_N2-m_N2)             \
                     + u_2Den * m_b2Den * e_n2Den   \
                     + u_3Den * m_b3Den * e_n3Den   \
                     + u_5Den * m_b5Den * e_n5Den \
                     + u_2DenC * m_b2DenC * e_n2Den  \
                     + u_3DenC * m_b3DenC * e_n3Den  \
                     + u_5DenC * m_b5DenC * e_n5Den \
                     + u_AOX * m_bAOX * e_n2AOX

        else:
            # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5, 54, 55, 56) (Add copio 1 aer + 6 den)
            ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                     - ammonia_ox                   \
                     - anammox_nh4                  \
                     - u_NOO * m_bNOO               \
                     - u_NOOan * m_bNOOan               \
                     + u_Het * m_bHet * (1./y_oHet - 1)    \
                     + ddt_NH4_Fac                         \
                     + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                     + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                     + u_3Den * m_b3Den * (1./y_n3Den - 1) \
                     + u_4Den * m_b4Den * (1./y_n4Den - 1) \
                     + u_5Den * m_b5Den * (1./y_n5Den - 1) \
                     + u_6Den * m_b6Den * (1./y_n6Den - 1) \
                     + u_7Den * m_b7Den * (1./y_n7Den_N2O - 1) \
                     + u_HetC * m_bHetC * (1./y_oHet - 1)    \
                     + u_1DenC * m_b1DenC * (1./y_n1Den - 1) \
                     + u_2DenC * m_b2DenC * (1./y_n2Den - 1) \
                     + u_3DenC * m_b3DenC * (1./y_n3Den - 1) \
                     + u_4DenC * m_b4DenC * (1./y_n4Den - 1) \
                     + u_5DenC * m_b5DenC * (1./y_n5Den - 1) \
                     + u_6DenC * m_b6DenC * (1./y_n6Den - 1) \
            # Dinitrogen gas (produced by 4, 5, 8, 55) (Add copio 3 den)
            ddt_N2 = dil * (in_N2-m_N2)             \
                     + u_2Den * m_b2Den * e_n2Den   \
                     + u_3Den * m_b3Den * e_n3Den   \
                     + u_5Den * m_b5Den * e_n5Den \
                     + u_7Den * m_b7Den * e_n7Den_N2O \
                     + u_2DenC * m_b2DenC * e_n2Den  \
                     + u_3DenC * m_b3DenC * e_n3Den  \
                     + u_5DenC * m_b5DenC * e_n5Den \
                     + u_AOX * m_bAOX * e_n2AOX

        # N2O gas (produced by 54 and 56) (consumed by 55) (Add copio 3 den)
        ddt_N2O = dil * (in_N2O-m_N2O)                  \
                 + u_4Den * m_b4Den * e_n4Den   \
                 + u_6Den * m_b6Den * e_n6Den   \
                 + u_4DenC * m_b4DenC * e_n4Den   \
                 + u_6DenC * m_b6DenC * e_n6Den   \
                 - den_N2OtoN2
        
        if N2Oammonia == 0:
            ddt_N2O = dil * (in_N2O-m_N2O)          \
                     + u_4Den * m_b4Den * e_n4Den   \
                     + u_6Den * m_b6Den * e_n6Den   \
                     + u_4DenC * m_b4DenC * e_n4Den   \
                     + u_6DenC * m_b6DenC * e_n6Den   \
                     - den_N2OtoN2
        else:
            if N2Oammonia == 1:
                ddt_N2O = dil * (in_N2O-m_N2O)          \
                         + u_4Den * m_b4Den * e_n4Den   \
                         + u_6Den * m_b6Den * e_n6Den   \
                         + u_4DenC * m_b4DenC * e_n4Den   \
                         + u_6DenC * m_b6DenC * e_n6Den   \
                         + N2OammoniaRate \
                         - den_N2OtoN2         
       
         
        # Biomass of aerobic heterotrophs
        ddt_bHet = dil * (-m_bHet)              \
                   + u_Het * m_bHet 
        # Biomass of facultative heterotrophs
        ddt_bFac = dil * (-m_bFac)              \
                   + u_Fac * m_bFac        
        # Biomass of nitrate denitrifiers
        ddt_b1Den = dil * (-m_b1Den)            \
                   + u_1Den * m_b1Den     
        # Biomass of nitrite denitrifiers
        ddt_b2Den = dil * (-m_b2Den)            \
                   + u_2Den * m_b2Den       
        # Biomass of full denitrifiers
        ddt_b3Den = dil * (-m_b3Den)            \
                   + u_3Den * m_b3Den 
        # Biomass of NO2-->N2O
        ddt_b4Den = dil * (-m_b4Den)            \
                   + u_4Den * m_b4Den 
        # Biomass of N2O-->N2
        ddt_b5Den = dil * (-m_b5Den)            \
                   + u_5Den * m_b5Den          
        # Biomass of NO3-->N2O
        ddt_b6Den = dil * (-m_b6Den)            \
                   + u_6Den * m_b6Den 
        # Biomass of bookend NO3-->NO2, N2O-->N2
        ddt_b7Den = dil * (-m_b7Den)            \
                   + u_7Den * m_b7Den       
    # (Add copi) 
        # Biomass of aerobic heterotrophs
        ddt_bHetC = dil * (-m_bHetC)          \
                    + u_HetC * m_bHetC    
        # Biomass of nitrate denitrifiers
        ddt_b1DenC = dil * (-m_b1DenC)            \
                   + u_1DenC * m_b1DenC     
        # Biomass of nitrite denitrifiers
        ddt_b2DenC = dil * (-m_b2DenC)            \
                   + u_2DenC * m_b2DenC       
        # Biomass of full denitrifiers
        ddt_b3DenC = dil * (-m_b3DenC)            \
                   + u_3DenC * m_b3DenC 
        # Biomass of NO2-->N2O
        ddt_b4DenC = dil * (-m_b4DenC)            \
                   + u_4DenC * m_b4DenC 
        # Biomass of N2O-->N2
        ddt_b5DenC = dil * (-m_b5DenC)            \
                   + u_5DenC * m_b5DenC          
        # Biomass of NO3-->N2O
        ddt_b6DenC = dil * (-m_b6DenC)            \
                   + u_6DenC * m_b6DenC               
    # end
        # Biomass of AOA
        ddt_bAOO = dil * (-m_bAOO)              \
                   + u_AOO * m_bAOO       
        # Biomass of NOB
        ddt_bNOO = dil * (-m_bNOO)              \
                   + u_NOO * m_bNOO 
        # Biomass of anaerobic NOB
        ddt_bNOOan = dil * (-m_bNOOan)              \
                   + u_NOOan * m_bNOOan      
        # Biomass of anammox bacteria
        ddt_bAOX = dil * (-m_bAOX)              \
                   + u_AOX * m_bAOX 
        
        
        # apply changes to state variables normalized by timestep (dt = timesteps per day)
        m_Sd = m_Sd + ddt_Sd * dt
        m_Sp = m_Sp + ddt_Sp * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NO3 = m_NO3 + ddt_NO3 * dt
        m_NO2 = m_NO2 + ddt_NO2 * dt
        m_an = m_an + ddt_an * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_N2 = m_N2 + ddt_N2 * dt
        m_N2O = m_N2O + ddt_N2O * dt
        m_bHet = m_bHet + ddt_bHet * dt
        m_bFac = m_bFac + ddt_bFac * dt
        m_b1Den = m_b1Den + ddt_b1Den * dt
        m_b2Den = m_b2Den + ddt_b2Den * dt
        m_b3Den = m_b3Den + ddt_b3Den * dt
        m_b4Den = m_b4Den + ddt_b4Den * dt
        m_b5Den = m_b5Den + ddt_b5Den * dt
        m_b6Den = m_b6Den + ddt_b6Den * dt
        m_b7Den = m_b7Den + ddt_b7Den * dt
        # add Copio (1 aer + 6 den)
        m_bHetC = m_bHetC + ddt_bHetC * dt
        m_b1DenC = m_b1DenC + ddt_b1DenC * dt
        m_b2DenC = m_b2DenC + ddt_b2DenC * dt
        m_b3DenC = m_b3DenC + ddt_b3DenC * dt
        m_b4DenC = m_b4DenC + ddt_b4DenC * dt # add biomass of Den4
        m_b5DenC = m_b5DenC + ddt_b5DenC * dt # add biomass of Den5
        m_b6DenC = m_b6DenC + ddt_b6DenC * dt # add biomass of Den6
        # end
        m_bAOO = m_bAOO + ddt_bAOO * dt
        m_bNOO = m_bNOO + ddt_bNOO * dt
        m_bNOOan = m_bNOOan + ddt_bNOOan * dt
        m_bAOX = m_bAOX + ddt_bAOX * dt
        
        # pulse OM and O2 into the chemostat
        if (t % int((1/dt * pulse_int))) == 0:
            m_Sd = m_Sd + pulse_Sd
            m_bHet = m_bHet + pulse_bHet
            m_bFac = m_bFac + pulse_bFac
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_Sd[i] = m_Sd 
            out_Sp[i] = m_Sp
            out_O2[i] = m_O2
            out_NO3[i] = m_NO3
            out_NO2[i] = m_NO2 
            out_NH4[i] = m_NH4 
            out_N2O[i] = m_N2O
            out_N2[i] = m_N2
            out_bHet[i] = m_bHet
            out_bFac[i] = m_bFac
            out_b1Den[i] = m_b1Den
            out_b2Den[i] = m_b2Den
            out_b3Den[i] = m_b3Den
            out_b4Den[i] = m_b4Den
            out_b5Den[i] = m_b5Den
            out_b6Den[i] = m_b6Den
            #add copi (1 aerHet + 6 dens)
            out_bHetC[i] = m_bHetC
            out_b1DenC[i] = m_b1DenC
            out_b2DenC[i] = m_b2DenC
            out_b3DenC[i] = m_b3DenC
            out_b4DenC[i] = m_b4DenC
            out_b5DenC[i] = m_b5DenC
            out_b6DenC[i] = m_b6DenC
            # end
            out_bAOO[i] = m_bAOO
            out_bNOO[i] = m_bNOO
            out_bNOOan[i] = m_bNOOan
            out_bAOX[i] = m_bAOX
            out_uHet[i] = u_Het
            out_uFac[i] = u_Fac
            out_u1Den[i] = u_1Den
            out_u2Den[i] = u_2Den
            out_u3Den[i] = u_3Den 
            out_u4Den[i] = u_4Den
            out_u5Den[i] = u_5Den
            out_u6Den[i] = u_6Den
            #add copi (1 aerHet + 6 dens)
            out_uHetC[i] = u_HetC
            out_u1DenC[i] = u_1DenC
            out_u2DenC[i] = u_2DenC
            out_u3DenC[i] = u_3DenC 
            out_u4DenC[i] = u_4DenC
            out_u5DenC[i] = u_5DenC
            out_u6DenC[i] = u_6DenC           
            # end
            out_uAOO[i] = u_AOO
            out_uNOO[i] = u_NOO
            out_uNOOan[i] = u_NOOan
            out_uAOX[i] = u_AOX
            out_facaer[i] = np.nanmean(facaer)
            out_rHet[i] = heterotrophy
            out_rHetAer[i] = aer_heterotrophy
            out_rO2C[i] = oxy_consumption
            '''
            # Oxygen consumption rate
            oxy_consumption = u_Het * m_bHet / y_oO2    \
                              + ddt_O2_Fac              \
                              + u_AOO * m_bAOO / y_oAOO \
                              + u_NOO * m_bNOO / y_oNOO
             '''                 
            out_r1Den[i] = den_nar # nitrate uptake rate by two types of nitrate reducers (anaerobe, facultative)
            out_r2Den[i] = den_nir  # nitrite uptake rate by NO2-->N2 
            out_r3Den[i] = den_full # nitrate uptake rate by NO3-->N2
            out_r4Den[i] = den_NitritetoN2O  # nitrite uptake rate by NO2-->N2O
            out_r5Den[i] = den_N2OtoN2      # N2O uptake rate by N2O-->N2 
            out_r6Den[i] = den_NitratetoN2O  # nitrate uptake rate by NO3-->N2O
            out_rAOO[i] = ammonia_ox
            out_rNOO[i] = nitrite_ox
            out_rNOOan[i] = nitrite_ox_an
            out_rAOX[i] = anammox_nh4
            
            out_rN2Oammonia[i] = N2OammoniaRate
            
                        
            out_b7Den[i] = m_b7Den
            out_u7Den[i] = u_7Den
            
            
    return [out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2O, out_N2, \
            out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den, 
            out_bHetC, out_b1DenC, out_b2DenC, out_b3DenC, out_b4DenC, out_b5DenC, out_b6DenC,            
            out_bAOO, out_bNOO, out_bAOX, \
            out_uHet, out_uFac, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den,
            out_uHetC, out_u1DenC, out_u2DenC, out_u3DenC, out_u4DenC, out_u5DenC, out_u6DenC,
            out_uAOO, out_uNOO, out_uAOX, \
            out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rAOX, \
            out_rN2Oammonia, \
            out_bNOOan, out_uNOOan, out_rNOOan, out_b7Den, out_u7Den]
