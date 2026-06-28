# -*- coding: utf-8 -*-
"""
WriteFile: Functions for writing data to files
"""
import os as os
import numpy as np
import tqdm as tqdm
from decimal import Decimal

#%% File writing
def Write_Bitfiles(directory,origin ,results, overwrite_shots = False, shots = 0):
    if not os.path.exists(directory):
        os.makedirs(directory)
    N = np.size(results) # Results must be an array
    if type(results) == str:
        N = len(results)
    Nshots = N # Assume that the data are stored in data chunks
    if overwrite_shots: Nshots = shots
    filename = origin+f"-{Nshots:e}-0-.txt"
    path = directory+os.sep+filename
    # Check that the file does not already exist
    i = 1
    while (os.path.exists(path)) and i < 10**6:
        filename = origin+ f"-{Nshots:e}-{i}-.txt"
        path = directory+os.sep+filename
        i += 1
    try:
        with open(path,"a") as file:
            print("Saving Results...")
            if type(results) == str:
                file.write(results)
            else:
                for i in tqdm.tqdm(range(N)):
                    #print(str(bit))
                    file.write(results[i])
        print(f"\nRESULTS STORED IN: \n{path}\n")
    except:
        print(f"*****\nERROR: FILE {filename} HAS NOT BEEN CREATED IN:\n{path}\n*****")
        raise
        return "FILE_ERROR"
    return filename

def Combine_Bitfiles(inDir,outDir):
    """
    Combine_Bitfiles:
        Given a directory containing several bit files,
        create a new file by concatenating all the bits
        and save it in outDir using the appropriate filename format.
        The filename format is assumed to be obtained from the first file in inDir.
    """
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    fileList = Scan_Dir(path = inDir)
    nBits = 0 # Initialize the bit counter
    shots = 0
    name = "temp.txt"
    auxPath = outDir + os.sep + name
    with open(auxPath,"w") as outFile:
        for i in range(len(fileList)):
            with open(fileList[i],"r") as inFile:
                data = inFile.read()
                nData = len(data)
                nBits += nData

                auxShot = int(Decimal(os.path.basename(fileList[i]).split("--")[1].split("-")[1]))
                shots += auxShot
                outFile.write(data)

    # Rename the file using the expected format
    aux = os.path.basename(fileList[0])
    qc, rest = aux.split("--")
    qbits = rest.split('-')[0]
    origin = qc + "--" + qbits
    newName =  origin + f"-{shots:e}-0-.txt" # Assume the same number of qubits was used
    path = outDir +os.sep+newName
    # Check that the file does not already exist
    i = 1
    while (os.path.exists(path)) and i < 10**6:
        newName = origin+ f"-{shots:e}-{i}-.txt"
        path = outDir+os.sep+newName
        i += 1
    os.rename(auxPath,path)
    print(f"FILES IN DIRECTORY {inDir} WERE MERGED INTO FILE:\n {path}")
    print(f"Total Bits: {nBits}. Total shots: {shots}, qubits: {qbits}")
    return newName

def Delete_Directory(directory):
    """
    Remove the temporary directory
    """
    fileList = Scan_Dir(path = directory)
    for file in fileList:
        os.remove(file)
    os.rmdir(directory)
    return

def GetDirectory(path):
    words = path.split('/')[:-1]
    directory = '/'.join(words)
    return directory

def Write_Analysis_Files(QC, message, directory = "Data_Processing", showEndMessage = False):
    """
    Write the required message to the specified path.
    message corresponds to an array of strings that will be written with line breaks.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileName = f"{QC}_Results.txt"
    path = directory + os.sep + fileName
    with open(path, "a") as f:
        if np.size(message) == 1: # It is a single message, not an array
            f.write(message)
        else:
            for line in message:
                f.write(line)
                f.write("\n")
        f.write("\n")
    if showEndMessage:
        print(f"\nRESULTS STORED IN\n {path}")
    return

def Write_Distribution_Files(tArr,GArr, nameArr, directory):
    """
    TArr must be a list of arrays with the format
    [[tArr1],[tArr2],...]
    Same for GArr:
    [[GArr1],[GArr2],...]
    NameArr contains the filenames:
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
    nFiles = len(nameArr)
    for i in tqdm.tqdm(range(nFiles)):
        filePath = directory + os.sep + nameArr[i] + ".txt"
        nCoords = len(tArr[i])
        with open(filePath,"w") as file:
            for j in range(nCoords):
                file.write(f"{tArr[i][j]} \t {GArr[i][j]}\n")
    return

#%% File reading
def Scan_Dir(path= "Outputs"):
    """
    Collect all files from folders and subfolders.
    Assume that only .txt files are present.
    """
    fileList = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                filename, file_extension = os.path.splitext(entry.path)
                if file_extension == ".txt": # Only accept .txt files
                    fileList.append(entry.path)
            elif entry.is_dir():
                fileList = fileList + Scan_Dir(path = entry.path)
    return fileList

def ReadFile(filePath, readAsBit = False):
    if readAsBit:
        readType = "rb"
    else:
        readType = "r"
    try:
        with open(filePath,readType) as f:
            data = f.read()
    except:
        print(f"ERROR: THE FILE COULD NOT BE READ FROM PATH\n {filePath}")
        return None
    return data

def Load_Distributions(directory):
    """
    Load the stored KS distributions into memory.
    A directory is expected to contain:
        - Mixed section
        - Pr section
    """

    return

def LoadTimes(filePath):
    times = []
    bits = []
    with open(filePath) as file:
        line = file.readline()
        while line != "":
            sides = line.split(":") # Split the line into left and right parts
            times.append(float(sides[1]))
            compData = sides[0].split("--")[1] # Remove the computer name
            aux = compData.split("-") # Split into individual elements
            bits.append(float(aux[0])*float(aux[1]))
            line = file.readline()
    times = np.array(times,dtype = float)
    bits = np.array(bits,dtype = int)
    return times,bits

