# Improved by Lucas Hernandez Bellon (2026)
import numpy as np
from scipy.special import erfc as erfc
from scipy.special import gammaincc as gammaincc

class FrequencyTest:

    @staticmethod
    def monobit_test(binary_data:str, verbose=False):
        """
        The focus of the test is the proportion of zeroes and ones for the entire sequence.
        The purpose of this test is to determine whether the number of ones and zeros in a sequence are approximately
        the same as would be expected for a truly random sequence. The test assesses the closeness of the fraction of
        ones to 陆, that is, the number of ones and zeroes in a sequence should be about the same.
        All subsequent tests depend on the passing of this test.

        if p_value < 0.01, then conclude that the sequence is non-random (return False).
        Otherwise, conclude that the the sequence is random (return True).

        :param      binary_data         The seuqnce of bit being tested
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)

        """
        length_of_bit_string = len(binary_data)
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        # Variable for S(n)
        ones = np.sum(binArr, dtype = np.int64)  # Counting ones is the same as summing the bit sequence 
        count = 2*ones - length_of_bit_string # ones - (length_of_bit_string - ones)

        # Compute the test statistic
        sObs = count / np.sqrt(length_of_bit_string)

        # Compute p-Value
        p_value = erfc(abs(sObs) / np.sqrt(2.))

        if verbose:
            print('Frequency Test (Monobit Test) DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_bit_string)
            print('\t# of \'0\':\t\t\t', length_of_bit_string-ones)
            print('\t# of \'1\':\t\t\t', ones)
            print('\tS(n):\t\t\t\t', count)
            print('\tsObs:\t\t\t\t', sObs)
            print('\tf:\t\t\t\t\t', abs(sObs) / np.sqrt(2.))
            print('\tP-Value:\t\t\t', p_value)
            print('DEBUG END.')

        # return a p_value and randomness result
        return (p_value, (p_value >= 0.01))

    @staticmethod
    def block_frequency(binary_data:str, block_size=128, verbose=False):
        """
        The focus of the test is the proportion of ones within M-bit blocks.
        The purpose of this test is to determine whether the frequency of ones in an M-bit block is approximately M/2,
        as would be expected under an assumption of randomness.
        For block size M=1, this test degenerates to test 1, the Frequency (Monobit) test.

        :param      block_size:         The sequence of bits being tested
        :param      binary_data:        The length of each block
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)
        """

        length_of_bit_string = len(binary_data)

        if length_of_bit_string < block_size:
            block_size = length_of_bit_string

        # Compute the number of blocks based on the input given.  Discard the remainder
        number_of_blocks = length_of_bit_string // block_size

        if number_of_blocks == 1:
            # For block size M=1, this test degenerates to test 1, the Frequency (Monobit) test.
            return FrequencyTest.monobit_test(binary_data[0:block_size])
       
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        # Initialized variables. Now we vectorize instead of using a bucle
        blocks = binArr[:block_size*number_of_blocks].reshape((number_of_blocks,block_size)) # Store values in a matrix of blocks
        piArr = np.sum(blocks,axis = 1, dtype = np.int64)/block_size # np.sum represents the count of ones in each block (thats why it is axis 1)

        # Create a for loop to process each block
        proportion_sum = np.sum((piArr-0.5)**2)
        # Compute 4M Σ(πi -½)^2.
        result = 4.0 * block_size * proportion_sum

        # Compute P-Value
        p_value = gammaincc(number_of_blocks / 2, result / 2)

        if verbose:
            print('Frequency Test (Block Frequency Test) DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_bit_string)
            print("\tSize of Block:\t\t", block_size)
            print('\tNumber of Blocks:\t', number_of_blocks)
            print('\tCHI Squared:\t\t', result)
            print('\t1st:\t\t\t\t', number_of_blocks / 2)
            print('\t2nd:\t\t\t\t', result / 2)
            print('\tP-Value:\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value >= 0.01))