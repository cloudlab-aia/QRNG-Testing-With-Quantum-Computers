# -*- coding: utf-8 -*-
"""
PRNGs Testing
"""
import Analyse_Data as AD
import WriteFile as WF
from tqdm import tqdm
import os as os

dataDir = "badPRNG_examples/Randu"
PRNG = "Randu"
fileList = WF.Scan_Dir(dataDir)
outDir = "NIST_Results"
for i in tqdm(range(len(fileList))):
    filepath = fileList[i]
    file = os.path.basename(filepath)
    WF.Write_Analysis_Files(PRNG, f"ANALYSIS OF {file}:",directory=outDir,showEndMessage=False)
    
    Hmin = AD.Estimate_Hmin(filepath)
    
    H = AD.Estimate_H(filepath)
    
    WF.Write_Analysis_Files(PRNG, ["\t--> Entropy Estimations:", f"\t\t --> H_min = {Hmin} bits", f"\t\t --> H = {H} bits"],directory=outDir, showEndMessage= False)
    
    minC,d,probD,passed = AD.Estimate_KComplexity(filepath)
    WF.Write_Analysis_Files(PRNG,["\t --> Complexity Estimations:",f"\t\t --> Minimum Complexity = {minC} bits", f"\t \t --> Deficiency Function = {d}",
                               f"\t \t --> Probability of that deficiency with Randomness Hypothesis = {probD}", f"\t \t --> Test Passed: {str(passed)}"],directory=outDir,showEndMessage=False)
    AD.NIST_Battery(PRNG,filepath,outDir=outDir)