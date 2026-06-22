# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:34:28 2022

Purpose
-------
    Calculate the maximum rate that O2 can be diffused into a cell based on its traits

@author: Pearse Buchanan and Xin Sun
"""

import numpy as np

def po_coef(diam, Qc, C_to_N):
    '''
    Calculates the maximum rate that O2 can be diffused into a cell

    Parameters
    ----------
    diam : Float
        equivalent spherical diatmeter of the microbe (um)
    Qc : Float
        carbon quota of a single cell (mol C / um^3)
    C_to_N : Float
        carbon to nitrogen ratio of the microbes biomass
    
    Returns
    -------
    po_coef : Float
        The maximum rate of O2 diffusion into the cell in units of m3 / mmol-biomassN / day

    '''
    dc = 1.5776 * 1e-5      # diffusion coefficiency of O2; cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
    dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
    
    Vc = 4/3 * np.pi * (diam/2)**3  # volume of cell (um^3)
    Qn = Qc / C_to_N * Vc           # mol N / cell
    
    p1 = 4 * np.pi * dc * (diam*1e-6/2.0)   # convert diameter in um to meters because diffusion coefficient in is m^2/day
    pm = p1 / Qn  # m3/d/mol-Nbiomass/cell
    po_coef = pm * 1e-3  # m3/d/mmol-biomassN
    return po_coef


def pn2o_coef(diam, Qc, C_to_N):
    '''
    Calculates the maximum rate that N2O can be diffused into a cell

    Parameters
    ----------
    diam : Float
        equivalent spherical diatmeter of the microbe (um)
    Qc : Float
        carbon quota of a single cell (mol C / um^3)
    C_to_N : Float
        carbon to nitrogen ratio of the microbes biomass
    
    Returns
    -------
    pn2o_coef : Float
        The maximum rate of N2O diffusion into the cell in units of m3 / mmol-biomassN / day

    '''
    dc_n2o = 1.5776 * 1e-5 * 1.0049 # diffusion coefficiency of N2O = dc of o2 * 1.0049; 
                                    # cm^2/s for 12C, 35psu, 50bar, ref: https://unisense.com/wp-content/uploads/2021/10/Seawater-Gases-table.pdf
    dc_n2o = dc_n2o * 1e-4 * 86400  # cm^2/s --> m^2/day
    
    Vc = 4/3 * np.pi * (diam/2)**3  # volume of cell (um^3)
    Qn = Qc / C_to_N * Vc           # mol-biomassN / cell (Qc in traits file is (mol C / um^3))
    
    p1 = 4 * np.pi * dc_n2o * (diam*1e-6/2.0)   # m3/d  convert diameter in um to m because diffusion coefficient in is m^2/day
    pm = p1 / Qn   # m3/d/mol-N
    pn2o_coef = pm * 1e-3   # m3/d/mmol-biomassN
    # multiply this by N-N2O (mmol/m3), get mol N-N2O per mol-biomassN per day
    # if then multiply the above by y (mol N biomass / mol N-N2O) to get growth rate (1/d)
    return pn2o_coef
