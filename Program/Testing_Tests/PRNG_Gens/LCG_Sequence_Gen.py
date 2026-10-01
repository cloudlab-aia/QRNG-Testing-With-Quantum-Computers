# -*- coding: utf-8 -*-
"""
LCG sequence generator
"""

import numpy as np
import WriteFile as WF
from tqdm import tqdm

def LCG(size,a,c,m,seed=None,):
    """
    Linear Congruential Generator
    
    m is best if its a power of 2 
    """
    if seed == None:
        seed = np.random.randint(0,m-1)
        
    # Lets get the # of bits neccesary for represent the integers
    # Integers will be from 0 to m-1 # thanks to mod m
    bits_perWord = (m-1).bit_length()
    
    currX = seed
    binArr = np.zeros(size,dtype = str)
    
    """
    The idea:
        1. We displace the MSB to rightmost position by displacement >> operator
        2. We substract the bit by &1 AND operation
    """
    binArr[0] = (seed >> (bits_perWord-1))&1 # (Using the most significant bit. Knuth Vol 2 Sec 3.4.1 A)
    
    for i in range(1,size):
        newX = (a*currX + c)%m # Lcg expression
        currX = newX
        binArr[i] = (newX >> (bits_perWord-1))&1
    bitstring = "".join(binArr)
    return bitstring

nBits = 10**7
nFiles = 100

# Note that the ideal configuration is with Hull-Dobel Theorem
a = 1103515244 # 1103515245 # coprime of m for best period
c = 12345
m = 2**20-1
mstr = f"{m:2e}" 
for i in tqdm(range(nFiles)):
    WF.Write_Bitfiles(f"badPRNG_examples/LCG_{a}_{c}_{mstr}","lcg",LCG(nBits,a,c,m))