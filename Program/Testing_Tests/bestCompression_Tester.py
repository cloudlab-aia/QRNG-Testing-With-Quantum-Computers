# -*- coding: utf-8 -*-
"""
Coger algoritmos de:
    https://www.mattmahoney.net/dc/text.html

Nos falta un algoritmo de compresion basado en content mixing

Estamos usando https://github.com/ElsevierSoftwareX/SOFTX-D-24-00288

"""
import numpy as np
import WriteFile as WF
from tqdm import tqdm
from datetime import datetime
import os as os
import matplotlib.pyplot as plt

import gzip as gzip # Compresor de Huffmann https://docs.python.org/3/library/gzip.html
import lzip as lzip # Compresor de Cadenas de Markov https://www.nongnu.org/lzip/
import pyppmd as ppm # Compresor PPM https://pyppmd.readthedocs.io/en/latest/api_guide.html
import brotli as brotli # Otra mezcla de Huffmann y Lzip
import zpaq as zpaq # https://github.com/zen-ham/zpaq
import lzma as lzma # Python Native

import psutil as psutil # To test PPM with different memory sizes

compressorsArr = np.array(["gzip","lzip","PPM","Brotli","Zpaq","Lzma"])
nCompress = len(compressorsArr)
currCompressorslen = np.zeros(nCompress)
compressorsCount = np.zeros(nCompress) # We are using 6 different compressors

#%% Functions
def Check_Best_Compressor(string):
    """
    Dado un string (idealmente bytes), calcula su complejidad de Kolmogorov asociada
    """
    # Unnecessary assignation, but it is left as it is for debugging purposes
    huffmann = gzip.compress(string, compresslevel = 9) # Pasamos de bytes a bits
    lempel = lzip.compress_to_buffer(string,level = 9) # Lempel-Ziv
    PPM = ppm.compress(string, max_order = 64) # PPM. maxorder = 64 is maximum compression
    brot = brotli.compress(string, quality = 11) # 11 is max compression
    zp = zpaq.compress(string,level=5)
    lz = lzma.compress(string,preset=9 | lzma.PRESET_EXTREME) # Preset_Extreme prioritizes compression
    
    
    currCompressorslen[0] = len(huffmann) # Pasamos de bytes a bits
    currCompressorslen[1] = len(lempel) # Lempel-Ziv
    currCompressorslen[2] = len(PPM) # PPM. maxorder = 64 is maximum compression
    currCompressorslen[3] = len(brot) # Seems to be the best of them
    currCompressorslen[4] = len(zp)
    currCompressorslen[5] = len(lz)
    
    minIndex = np.argmin(currCompressorslen)
    compressorsCount[minIndex] += 1
    K = currCompressorslen[minIndex] # Kolmogorov Complexity converges to the best compression possible
    return K*8 # We did the comparison in bytes, but Kolmogorov Complexity has to be in bits

def Test_PPM_Configs(string):
    """
    You can check that with any config the compression rate is the same--> PPM isnt good for this task
    """
    m = np.arange(0.1,0.7,0.1) # Memory relative size
    o = np.arange(2,65,1) # Max order 
    PPMArr = np.zeros((len(m),len(o))) # Matrix of (m,o) cases
    for i in range(len(m)):
        for j in range(len(o)):
            PPMArr[i,j] = len(ppm.compress(string, max_order = o[j],mem_size = psutil.virtual_memory().available*m[i]))*8
    return PPMArr

#%% Main Variables
dirList = ["badPRNG_examples/LCG_1103515244_12345_1.048575e+06"]


#%% Main execution
filesList = []
for x in dirList:    
    filesList = WF.Scan_Dir(x)
nFiles = len(filesList)
KArr = np.ones(nFiles) * (-1) # -1 for sentry purposes
NArr = np.ones(nFiles) * (-1)
print(f"Testing which compressor performs best for files in\n{dirList}\n...")

for i in tqdm(range(nFiles)):
    bitstring = WF.ReadFile(filesList[i],readAsBytes=True)
    """
    Currently bitstring is formatted with ASCII (each "0" corresponds to 00110000 bit, etc)
    This correspondence made by ASCII affects compression rate (as they are not optimally compression algorithms)
    thus we transform the sequences to their bytes sequence equivalent
    """
    # Remove bits that are insufficient to pack into bytes
    nBytes = len(bitstring)//8
    #print("Bits eliminados:", len(bitstring)%8)
    # Translate to bytes
    bytestring = bytearray()
    for b in range(0,nBytes):
        bytestring.append(int(bitstring[8*b:8*(b+1)],2))
        
    bytestring = bytes(bytestring) # Ya no vamos a annadir mas bytes  
    NArr[i] = len(bytestring)*8 # Len on bits instead of bytes
    KArr[i] = Check_Best_Compressor(bytestring)

        
bestIndex = np.argmax(compressorsCount)
print("-"*40)
print("Compression Done!")
print(f"Best Algorithm was {compressorsArr[bestIndex]}, having the best compression {compressorsCount/nFiles*100} % of the times")
print()
print("\nAll Results are the following:")
for i in range(nCompress):
    print(f"\t # of Compressor {compressorsArr[i]} best performance: {compressorsCount[i]} out of {nFiles}  ({compressorsCount[i]/nFiles*100} %")

fileResultsName = "Compression_Results/Summary.txt"
with open(fileResultsName,"w") as file:
    file.write(f"Tested at {datetime.now()}\n")
    file.write("BitFiles from:\n")
    for i in range(len(dirList)):
        file.write(f"\t {dirList[i]}\n")
    file.write(f"Total Files tested: {nFiles}\n\n")
        
    file.write("Compressors used:\n")
    for i in range(nCompress):
        file.write(f"\t {compressorsArr[i]}: {compressorsCount[i]} out of {nFiles} was best compressor ({compressorsCount[i]/nFiles*100}) %\n")
    file.write("\n\n")
    file.write(f"Best Compressor: {compressorsArr[bestIndex]} with {compressorsCount[bestIndex]/nFiles*100} % of success")
print(f"Results saved at {fileResultsName}")

print("Generating p-values of the test...")
dArr = NArr-KArr # Deficiency Function
dArr = np.where(dArr >= 0, dArr,0)

pArr = 2.0**(-dArr)
print("Saving Results...")
pValuesFile = "Compression_Results/pValues.txt"
with open(pValuesFile, "w") as file:
    count = 0
    file.write("File \t # of File \t pvalue\n")
    for i in range(nFiles):
        file.write(f"{os.path.basename(filesList[i])} \t {count} \t {pArr[i]}\n ")
        count +=1
print(f"pvalues saved in {pValuesFile}")
# Create an historgram
nBins = 10**2
fig,ax = plt.subplots()
fig.suptitle("pvalue Histogram of Compression Test")
ax.set_xlabel("pValues")
ax.set_xlim(0,1.01)
ax.set_ylabel("Counts")
ax.hist(pArr,bins=nBins)
fig.savefig("Compression_Results/histogram.jpg",dpi = fig.dpi, bbox_inches='tight')
print("\n\nTEST DONE")
