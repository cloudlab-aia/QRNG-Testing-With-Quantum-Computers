#Improved by Lucas Hernandez Bellon (2026)
import numpy as np
from scipy.special import erfc as erfc

class Universal:
    # Moved to class variable to avoid reinitialization of them
    variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
    expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
    @staticmethod
    def statistical_test(binary_data:str, verbose=False):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of this test is the number of bits between matching patterns (a measure that is related to the
        length of a compressed sequence). The purpose of the test is to detect whether or not the sequence can be
        significantly compressed without loss of information. A significantly compressible sequence is considered
        to be non-random. **This test is always skipped because the requirements on the lengths of the binary
        strings are too high i.e. there have not been enough trading days to meet the requirements.

        :param      binary_data:    a binary string
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)
        """
        length_of_binary_data = len(binary_data)
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
    
        if length_of_binary_data >= 1059061760:
            pattern_size = 16 
        elif length_of_binary_data >= 496435200:
            pattern_size = 15
        elif length_of_binary_data >= 231669760:
            pattern_size = 14
        elif length_of_binary_data >= 107560960:
            pattern_size = 13
        elif length_of_binary_data >= 49643520:
            pattern_size = 12
        elif length_of_binary_data >= 22753280:
            pattern_size = 11
        elif length_of_binary_data >= 10342400:
            pattern_size = 10
        elif length_of_binary_data >= 4654080:
            pattern_size = 9
        elif length_of_binary_data >= 2068480:
            pattern_size = 8
        elif length_of_binary_data >= 904960:
            pattern_size = 7
        elif length_of_binary_data >= 387840:
            pattern_size = 6 
        else:
            return (-1.0, False)
        
        # Keeps track of the blocks, and whether were are initializing or summing
        num_blocks = length_of_binary_data // pattern_size
        # Q = 10 * pow(2, pattern_size)
        init_bits = 10 * 2**pattern_size

        test_bits = num_blocks - init_bits
        
        # Examine each of the K blocks in the test segment and determine the number of blocks since the
        # last occurrence of the same L-bit block (i.e., i – Tj). Replace the value in the table with the
        # location of the current block (i.e., Tj= i). Add the calculated distance between re-occurrences of
        # the same L-bit block to an accumulating log2 sum of all the differences detected in the K blocks
        
        """
        We now do something similar to the new Approximate Entropy Test to get binary patterns: 
            pattArr stores in integer form which pattern of size pattern_size appears block by block
            Vobs stores the # of appearance of different combinations 
            of substrings of size pattern_size (indicated in their integer form by array index)
            
            Vobs now is calculated by pattArr
        """
        # We create pattArr
        pattArr = np.zeros(num_blocks, dtype = np.uint32)
        # We copy the patterns of size m+1 of the bitstring in periodic conditions bit by bit
        for i in range(pattern_size): # CANT BE PARALLELIZED, as order is important due to the displacement <<1
            pattArr <<= 1 # Make space for the next bit
            pattArr |= binArr[i:pattern_size*num_blocks:pattern_size]
        
        # Now that pattArr is done, we can count directly
        index_order = np.argsort(pattArr, kind="stable") # We use "stable" so that in draws they are saved by order in pattArr
        pattOrdered = pattArr[index_order]
        distArr = index_order + 1 # We initialize the distances with the positions +1, so that the first positions of the patterns dont return 0
        clash = pattOrdered[1:] == pattOrdered[:-1] # Mask that marks where there is the same pattern in following positions
        distArr[1:] = np.where(clash,index_order[1:]-index_order[:-1],index_order[1:]+1) # In clash zones we compute the distance. If there is no clash there the distance is incremented by 1
        
        cumsum = np.sum(np.log2(distArr[index_order >= init_bits])) # We only sum in the test block, discarding distances computed in the test zone
        # These are the expected values assuming randomness (uniform)
        c = 0.7 - 0.8 / pattern_size + (4. + 32. / pattern_size) * test_bits**(-3. / pattern_size) / 15.
        sigma = c * np.sqrt(Universal.variance[pattern_size] / test_bits)

        # Compute the statistic     
        phi = cumsum / test_bits
        stat = abs(phi - Universal.expected[pattern_size]) / (np.sqrt(2.) * sigma)

        # Compute for P-Value
        p_value = erfc(stat)

        if verbose:
            print('Maurer\'s Universal Statistical Test DEBUG BEGIN:')
            print("\tLength of input:\t\t", length_of_binary_data)
            print('\tLength of each block:\t', pattern_size)
            print('\tNumber of Blocks:\t\t', init_bits)
            print('\tValue of phi:\t\t\t', phi)
            print('\tP-Value:\t\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value>=0.01))
