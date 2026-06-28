# -*- coding: utf-8 -*-
"""
Plot Distributions
We plot the distributions of p-values of each NIST test for IBM and PRNG
"""
import WriteFile as WF
import numpy as np

#%% Inputs
testsNames = ["Monobit Test","Frequency Within a Block Test", "Run Test", "Longest One Block Test",
              "Spectral Test", "Approximate Entropy Test", "Cusum Test", "Maurer Test", "Binary Matrix Test",
              "Linear Complexity Test"] # A parte Variant y Run Test
dirIBM = "KS_Results/Distributions_IBM"
dirSim = "KS_Results/Distributions_Sim"
#%% Procesado
nTests = len(testsNames)
GprIBM = np.zeros(nTests,dtype = object)
GprSim = np.zeros(nTests,dtype = object)
GmixIBM = np.zeros(nTests,dtype = object)
GmixSim = np.zeros(nTests,dtype = object)

# Llamamos a los datos
for i in range(nTests):
    GprIBM[i] = WF.Load_Distributions()
#%% Plots 

