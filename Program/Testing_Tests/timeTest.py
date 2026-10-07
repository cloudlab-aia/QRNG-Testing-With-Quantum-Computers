# -*- coding: utf-8 -*-
"""
Time Tester
Tests NIST Tests' times
"""

from time import time
import numpy as np
import WriteFile as WF
from tqdm import tqdm

# Tests NIST (by stevenang: https://github.com/stevenang/randomness_testsuite)
from NIST.ApproximateEntropy import ApproximateEntropy as aet
from NIST.Complexity import ComplexityTest as ct
from NIST.CumulativeSum import CumulativeSums as cst
from NIST.FrequencyTest import FrequencyTest as ft
from NIST.RandomExcursions import RandomExcursions as ret
from NIST.RunTest import RunTest as rt
from NIST.Spectral import SpectralTest as st
from NIST.Universal import Universal as ut

from Analyse_Data import Binary_Matrix_Test
from Analyse_Data import Block_Frequency_Test

# TESTS NIST IMPROVED
from NIST_Improved.Universal import Universal as utI
from NIST_Improved.Complexity import ComplexityTest as ctI
from NIST_Improved.ApproximateEntropy import ApproximateEntropy as aetI
from NIST_Improved.CumulativeSum import CumulativeSums as cstI
from NIST_Improved.RandomExcursions import RandomExcursions as retI
from NIST_Improved.RunTest import RunTest as rtI
from NIST_Improved.Spectral import SpectralTest as stI
from NIST_Improved.FrequencyTest import FrequencyTest as ftI
#%% Variables

testsNames = ["Monobit Test","Frequency Test Within a Block", "Runs Test", "Test for the Longest Run of Ones in a Block",
              "Spectral Test", "Approximate Entropy Test", "Cusum Test", "Maurer's Test", "Binary Matrix Rank Test",
              "Linear Complexity Test"] # Random Excursions can be activated via using_RandomExcursion_Tests Variable
tests = [ftI.monobit_test,Block_Frequency_Test,rtI.run_test, rtI.longest_one_block_test,stI.spectral_test,aetI.approximate_entropy_test,
         cstI.cumulative_sums_test,utI.statistical_test,
         Binary_Matrix_Test,ctI.linear_complexity_test]


filePath = "Randu/randu-1.000000e+07-0-.txt"
savePath = "Time_Results.txt"
debug = False
using_RandomExcursion_Tests = True
#%% Main execution
bitstring = WF.ReadFile(filePath)

nTests = len(tests)
timesArr = np.zeros(nTests + 1, dtype = np.float64) # 0 corresponds to program start.
pArr = np.zeros(nTests)

if using_RandomExcursion_Tests:
    timesArr = np.zeros(nTests + 3, dtype = np.float64) # 0 corresponds to program start.
    testsNames += ["Random Excursions Test","Random Excursions Variant Test"]
    
    timesArr[0] = time()
    for i in tqdm(range(nTests)):
        pArr[i] = tests[i](bitstring, verbose=debug)[0] # [0] only for normal tests
        timesArr[i+1] = time()
        #print(pArr[i])
    pRD = retI.random_excursions_test(bitstring,verbose=debug)
    timesArr[-2] = time()
    pVar = retI.variant_test(bitstring,verbose=debug)
    timesArr[-1] = time() 
    nTests += 2
  
else:
    timesArr[0] = time()
    for i in tqdm(range(nTests)):
        pArr[i] = tests[i](bitstring, verbose=debug)[0] # [0] only for normal tests
        timesArr[i+1] = time()
        #print(pArr[i])

# Save Timestamps
with open(savePath,"w") as f:
    f.write(f"TIME RESULTS OF NIST's BATTERY for file {filePath}:\n")
    for i in range(nTests):
        f.write(f"{testsNames[i]}: {timesArr[i+1]-timesArr[i]}\n")
        
    sort = np.argsort(timesArr[1:]-timesArr[:-1])
    f.write("Sorted by least to most expensive:\n")
    for j in range(nTests):
        f.write(f"{testsNames[sort[j]]} < ")