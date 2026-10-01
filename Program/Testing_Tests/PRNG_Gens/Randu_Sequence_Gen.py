# -*- coding: utf-8 -*-
"""
https://github.com/ptomato/randu/tree/master
"""

from randu import Randu
from datetime import datetime
import WriteFile as WF

seed = datetime.now()
nBits = 10**7
nFiles = 100 
gen = Randu(seed) # I use the same seed for all files to check PRNG period
print(f"Generating {nFiles} sequences of {nBits} with Randu flawed PRNG...")
for i in range(nFiles):
    WF.Write_Bitfiles("badPRNG_examples/Randu", "randu", gen.choices(["0","1"], k = nBits))
print("Done!\n\n")

