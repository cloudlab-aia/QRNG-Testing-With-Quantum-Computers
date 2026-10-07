# Improvement made by Lucas Hernandez Bellon (2026)
from numpy import histogram as histogram
from scipy.special import gammaincc as gammaincc

import numpy as np
from numba import njit, prange # njit has automatically nopython=True

class ComplexityTest:
    
    def linear_complexity_test(binary_data:str, verbose=False, block_size=500): # May Increase block_size to 1000
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of this test is the length of a linear feedback shift register (LFSR). The purpose of this test is to
        determine whether or not the sequence is complex enough to be considered random. Random sequences are
        characterized by longer LFSRs. An LFSR that is too short implies non-randomness.

        :param      binary_data:    a binary string
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :param      block_size:     Size of the block (between 500 and 50000)
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)

        """

        # Convert binary_Data to numpy array for optimization purposes
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        
        degFreedom,xObs = ComplexityTest.Compute_xObs(binArr,verbose=verbose,block_size=block_size) # Separate frombuffer as it is not compatible with numba
        if degFreedom == -1.0: # Number of blocks <= 1. We cant do the test
            return (-1.0, False)
        
        # P-Value = igamc(K/2, xObs/2)
        p_value = gammaincc(degFreedom / 2.0, xObs / 2.0)
    
        if verbose:
            print('\tP-Value:\t\t\t', p_value)
            print('DEBUG END.')
    
        return (p_value, (p_value >= 0.01))

    @njit(parallel=True)    
    def Compute_xObs(binArr,verbose,block_size):    
        length_of_binary_data = len(binArr)
        # The number of degrees of freedom;
        # K = 6 has been hard coded into the test.
        degree_of_freedom = 6
    
        #  π0 = 0.010417, π1 = 0.03125, π2 = 0.125, π3 = 0.5, π4 = 0.25, π5 = 0.0625, π6 = 0.020833
        #  are the probabilities computed by the equations in Section 3.10
        pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
        mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - (block_size / 3.0 + 2.0 / 9) /(2.0 ** block_size) # Theoretical mean
        number_of_block = length_of_binary_data//block_size # dismisses some bits
        if number_of_block <= 1:
            return (-1.0, False)
        
        # Partition the n-bit sequence into N independent blocks of M bits -- Could be done with numpy crops
        blocks = binArr[:number_of_block*block_size].reshape(number_of_block, block_size)
        # Determination of Linear complexity of the block -- Li
        complexities = np.zeros(number_of_block,dtype = np.float64)
        for i in prange(number_of_block): # may parallelize with prange?
            #complexities[i] = berlekamp_massey_scalar(blocks[i])
            complexities[i] = berlekamp_massey_packed(blocks[i], block_size)
        # Calculations of Ti -- Vectorized
        t = -1.0 * (((-1) ** block_size) * (complexities - mean) + 2.0/9.)
        
        vg = histogram(t, bins=[-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf])[0][::-1] #vii in NIST docs
        xObs = np.sum(((vg - number_of_block * pi) ** 2) / (number_of_block * pi)) # Terms of the summatory of X^2 obs
        if verbose:
            print('Linear Complexity Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tLength in bits of a block:\t',block_size )
            print("\tDegree of Freedom:\t\t", degree_of_freedom)
            print('\tNumber of Blocks:\t', number_of_block)
            print('\tValue of Vs:\t\t', vg)
            print('\txObs:\t\t\t\t', xObs)
        return degree_of_freedom,xObs
    

@njit()
def berklekamp_massey_matrix(block_data):
    """
    Slightly more readable, but slightly more inefficient than packed version
    """
    """
    An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
    [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
    The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
    for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
    sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
    non-zero elements to have a multiplicative inverse.
    :param block_data:
    :return:
    """
    n = len(block_data)
    c = np.zeros(n+1,dtype = np.uint8)
    b = np.zeros(n+1,dtype = np.uint8)
    c[0], b[0] = 1, 1
    l, m = 0, -1
    for i in range(n):
        # Solving the Linear Equations System
        v = block_data[i - l:i][::-1]
        """
        multiplication in binary is equivalent to an AND (&), and the sum of the dot product can be done with np.sum
        (Could be even more efficient if full binary operations were used, check berlekamp_massey_packed)
        Lastly The sum in mod 2 is equivalent to an XOR ^
        That way: 
            d = (block_data[i] + dot(v, cc)) % 2
        Is now:
            d = block_data[i] ^ np.sum(v & c[1:l+1])%2
        """
        d = block_data[i] ^ (np.sum(v & c[1:l+1])%2)
        if d == 1:
            temp = np.copy(c)
            for j in range(l): # XOR of c (more efficient that the one made in the original script)
                if b[j] == 1:
                    c[i+j-m] ^= 1
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b = temp
    return l

@njit(cache=True)
def berlekamp_massey_packed(block_data, M): # M: Size of the blocks (block_size)
    W = M // 64 + 1                          # We treat the blocks as words 64 bits (including the last bit of block_data, thats the +1)
    c = np.zeros(W, dtype = np.uint64) # Coefficients of our current LFSR (like w, but in array form instead of integer)
    b = np.zeros(W, dtype = np.uint64) # LFSR that worked in steps i-1
    t = np.zeros(W, dtype = np.uint64) # Temporal buffer to interchange with b (in order to avoid creating new arrays inside the for)
    w = np.zeros(W, dtype = np.uint64) # Bitstring copied bit by bit while creating the LFSR
    
    c[0] = np.uint64(1) # Coded with unit64 for bit operations
    b[0] = np.uint64(1)
    l = 0 # length of current LFSR
    m = -1 # Position of the bit in block_data where the LFSR proposed failed
    for i in range(M): # In each bit of the block:
        # w: k bit is bits[i-k] we displace 1 to the left in order to insert s[i] --> We add the bit i of block_data to w
        for j in range(W - 1, 0, -1):
            w[j] = (w[j] << np.uint64(1)) | (w[j - 1] >> np.uint64(63))
        w[0] = (w[0] << np.uint64(1)) | np.uint64(block_data[i])

        # LFSR works when updating with a new bit  = paridad de (w & c)
        x = np.uint64(0) # 0 + 0*2^1 + ... + 0*2^64 --> The bitstring recreated with the LFSR of last step
        for j in range(W):
            x ^= w[j] & c[j] # We create what LFSR of last step would do 
        # Move bits to their respective positions
        x ^= x >> np.uint64(32)
        x ^= x >> np.uint64(16)
        x ^= x >> np.uint64(8)
        x ^= x >> np.uint64(4)
        x ^= x >> np.uint64(2)
        x ^= x >> np.uint64(1)

        if x & np.uint64(1): # There is a discrepancy: x doesnt recreate block_data correctly
            for j in range(W):
                t[j] = c[j]
            sh = i - m                       # c ^= b * x^(i-m)
            ws = sh >> 6
            bs = sh & 63
            for j in range(W - 1, ws - 1, -1):
                src = j - ws
                v = b[src] << np.uint64(bs)
                if bs > 0 and src > 0:
                    v |= b[src - 1] >> np.uint64(64 - bs)
                c[j] ^= v
            if 2 * l <= i:
                l = i + 1 - l
                m = i
                for j in range(W):
                    b[j] = t[j] # New b corresponds to the new LFSR created with c (already copied to t)
    return l