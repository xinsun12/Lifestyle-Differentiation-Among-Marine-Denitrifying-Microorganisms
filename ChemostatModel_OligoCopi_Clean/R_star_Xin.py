# -*- coding: utf-8 -*-
"""
Created in July 2022

Purpose
-------
    Calculate R* (subsistence concentration of nutrient for microbial population)
    (µM)
@author: Xin Sun
"""


def R_star(loss, K_r, Vmax, y_r):
    '''
    
    Parameters
    ----------
    loss : TYPE
        biomass loss rate = dilution rate in chemostat setting
    K_r : TYPE
        half-saturation coefficient for nutrient uptake
    Vmax : TYPE
        Maximum nutrient uptake rate
    y_r : TYPE
        Yield of biomass creation (mol BioN per mol compound)

    Returns
    -------
    TYPE
        R* = the subsistence concentration of the compound in question required to sustain biomass 

    '''
    return (loss * K_r) / (Vmax * y_r - loss)


def R_starMM2(loss, K1, mumax, S2, K2):
    '''
    
    Parameters
    ----------
    loss : TYPE
        biomass loss rate = dilution rate in chemostat setting
    K1 : TYPE
        half-saturation coefficient for nutrient uptake
    mumax : TYPE
        Maximum growth rate
    S2 : TYPE (must in µM)
        Equilibrium concentration of the other substrate in the double MM equation
    K2: TYPE (must in µM)
        half-saturation constant for the other substrate in the double MM equation

    Returns
    -------
    TYPE
        R* = the subsistence concentration of the compound in question required to sustain biomass 

    '''
    return (loss * K1) / (mumax * S2/(S2+K2)- loss)