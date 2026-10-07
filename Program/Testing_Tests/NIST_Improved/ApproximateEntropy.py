# Improvements made by Lucas Hernandez Bellon (2026)
import numpy as np
from scipy.special import gammaincc as gammaincc

class ApproximateEntropy:
    @staticmethod
    def approximate_entropy_test(binary_data:str, verbose=False, pattern_length=10):
        """
        from the NIST documentation http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf

        As with the Serial test of Section 2.11, the focus of this test is the frequency of all possible
        overlapping m-bit patterns across the entire sequence. The purpose of the test is to compare
        the frequency of overlapping blocks of two consecutive/adjacent lengths (m and m+1) against the
        expected result for a random sequence.

        :param      binary_data:        a binary string
        :param      verbose             True to display the debug message, False to turn off debug message
        :param      pattern_length:     the length of the pattern (m)
        :return:    ((p_value1, bool), (p_value2, bool)) A tuple which contain the p_value and result of serial_test(True or False)
        
        NOTE: In our case as data has 10^7 bits, NIST advises to use patter_length < 17 for ensuring good chi square approximation
        """
        if pattern_length > 32:
            pattern_length = 32 # We will use int32 in the container of the patterns, so that is the limit
        
        length_of_binary_data = len(binary_data)

        # Augment the n-bit sequence to create n overlapping m-bit sequences by appending m-1 bits
        # from the beginning of the sequence to the end of the sequence.
        # NOTE: documentation says m-1 bits but that doesnt make sense, or work.
        binary_data += binary_data[:pattern_length + 1:] # They do this to avoid index out of range during counting
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
    
        # Removed max pattern as now vobs are initialized counting card{0,1}^m
        """
        Changed the form of counting patterns of size pattern_length in the sequence:
            1. We get the bitstring in periodic conditions with an array uint8 (binArr)
            2. We create an auxilary array that stores which pattern of size m+1 appears in each window (in integer form) as:
                i. Create it empty --> pattArr = [0,0,0,0,0,...] --> PattArr has to be int32 for ensuring the following displacements dont erase the most significant bit IMPORTANT FOR STEP *
                ii. We start marking which pattern correspond to each index by bitwise copy:
                    a. Displacement: pattArr = [00x, 00x, ...]
                    b. copying thanks to an OR operation (the displacement generates a 0, so 0 | a = a)
                    pattArr |= bitArr[i:i+n] --> Vectorized, we do this m+1 times with displacements in order to store windows of size m+1 
            3. We count how many times each pattern has appeared with an histogram with bins separated by integers
            *. Patterns of size m can be calculated by displacing all bits of pattArr one position to the right (thus removing the last displacement)
        """
        
        # We create pattArr
        pattArr = np.zeros(length_of_binary_data, dtype = np.uint32)
        # We copy the patterns of size m+1 of the bitstring in periodic conditions bit by bit
        for i in range(pattern_length+1): # CANT BE PARALLELIZED, as order is important due to the displacement <<1
            pattArr <<= 1 # Make space for the next bit
            pattArr |= binArr[i:i+length_of_binary_data]
        
        # Now that we have the patterns stored, we count them
        # NOTE: bins store the count by index, so index i corresponds to the time that pattern with integer i has appeared
        p_01 = np.bincount(pattArr>>1, minlength = 2**(pattern_length))/length_of_binary_data # probability of each m-bit patterns
        p_02 = np.bincount(pattArr, minlength = 2**(pattern_length+1))/length_of_binary_data # probability of each m+1-bit patterns
            
        # Calculate the test statistics and p values
        # Removed vobs list
        sum_01 = np.sum(p_01*np.log(p_01/length_of_binary_data,where=p_01 > 0, out = np.zeros(2**pattern_length))) # Mask of 0s explicit in log 
        sum_02 = np.sum(p_02*np.log(p_02/length_of_binary_data,where=p_02 > 0, out = np.zeros(2**(pattern_length+1)))) # Mask of 0s explicit in log 
        ape = sum_01 - sum_02

        xObs = 2.0 * length_of_binary_data * (np.log(2) - ape)
        p_value = gammaincc(pow(2, pattern_length - 1), xObs / 2.0)
        if verbose:
            print('Approximate Entropy Test DEBUG BEGIN:')
            print("\tLength of input:\t\t\t", length_of_binary_data)
            print('\tLength of each block:\t\t', pattern_length)
            print('\tApEn(m):\t\t\t\t\t', ape)
            print('\txObs:\t\t\t\t\t\t', xObs)
            print('\tP-Value:\t\t\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value >= 0.01))
    