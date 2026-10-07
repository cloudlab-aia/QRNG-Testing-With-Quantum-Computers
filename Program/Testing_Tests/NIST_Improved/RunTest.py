# Improved by Lucas Hernandez Bellon (2026)
from scipy.special import erfc as erfc
from scipy.special import gammaincc as gammaincc
import numpy as np
from numba import njit

class RunTest:

    @staticmethod
    def run_test(binary_data:str, verbose=False):
        """
        The focus of this test is the total number of runs in the sequence,
        where a run is an uninterrupted sequence of identical bits.
        A run of length k consists of exactly k identical bits and is bounded before
        and after with a bit of the opposite value. The purpose of the runs test is to
        determine whether the number of runs of ones and zeros of various lengths is as
        expected for a random sequence. In particular, this test determines whether the
        oscillation between such zeros and ones is too fast or too slow.

        :param      binary_data:        The seuqnce of bit being tested
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)
        """

        length_of_binary_data = len(binary_data)
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        # Predefined tau = 2 / sqrt(n)
        tau = 2 / np.sqrt(length_of_binary_data) # Checked the todo of the original code: It is written right, example 2.3.8 of the documentation has an errata

        # Step 1 - Compute the pre-test proportion πof ones in the input sequence: π = Σjεj / n
        one_count = np.sum(binArr,dtype = np.int64)

        pi = one_count / length_of_binary_data

        # Step 2 - If it can be shown that absolute value of (π - 0.5) is greater than or equal to tau
        # then the run test need not be performed.
        if abs(pi - 0.5) >= tau:
            #print("The test should not have been run because of a failure to pass test 1, the Frequency (Monobit) test.")
            return (0.0000, False)
        
        # Step 3 - Compute vObs. Now it is vectorized by with a foward copy of the bitstring
        vObs = len(np.where(binArr[1:] != binArr[:-1])[0]) + 1 # +1 Is given by the formula on 2.3.4
        # Step 4 - Compute p_value = erfc((|vObs − 2nπ * (1−π)|)/(2 * sqrt(2n) * π * (1−π)))
        p_value = erfc(abs(vObs - (2 * (length_of_binary_data) * pi * (1 - pi))) / (2 * np.sqrt(2 * length_of_binary_data) * pi * (1 - pi)))

        if verbose:
            print('Run Test DEBUG BEGIN:')
            print("\tLength of input:\t\t\t\t", length_of_binary_data)
            print("\tTau (2/sqrt(length of input)):\t", tau)
            print('\t# of \'1\':\t\t\t\t\t\t', one_count)
            print('\t# of \'0\':\t\t\t\t\t\t', binary_data.count('0'))
            print('\tPI (1 count / length of input):\t', pi)
            print('\tvObs:\t\t\t\t\t\t\t', vObs)
            print('\tP-Value:\t\t\t\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value > 0.01))

    def longest_one_block_test(binary_data:str, verbose=False):
        """
        The focus of the test is the longest run of ones within M-bit blocks. The purpose of this test is to determine
        whether the length of the longest run of ones within the tested sequence is consistent with the length of the
        longest run of ones that would be expected in a random sequence. Note that an irregularity in the expected
        length of the longest run of ones implies that there is also an irregularity in the expected length of the
        longest run of zeroes. Therefore, only a test for ones is necessary.

        :param      binary_data:        The sequence of bits being tested
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)
        """
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        xObs,k = RunTest.Compute_xObs(binArr, verbose=verbose)
        if xObs == -1:
            return (-1,False)
        p_value = gammaincc(0.5*k , 0.5*xObs)
        if verbose:
            print('\tP-Value:\t\t\t\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value > 0.01))
    
    @njit()
    def Compute_xObs(binArr, verbose = False):
        length_of_binary_data = len(binArr)
        # Initialized k, m. n, pi and v_values
        if length_of_binary_data < 128:
            # Not enough data to run this test
            return (-1, False)
        elif length_of_binary_data < 6272:
            k = 3
            m = 8
            v_values = np.arange(1,5,1)
            pi_values = np.array([0.21484375, 0.3671875, 0.23046875, 0.1875])
        elif length_of_binary_data < 750000:
            k = 5
            m = 128
            v_values = np.arange(4,10,1)
            pi_values = np.array([0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847])
        else:
            # If length_of_bit_string > 750000
            k = 6
            m = 10000
            v_values = np.arange(10,17,1)
            pi_values = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])

        number_of_blocks = length_of_binary_data // m
        frequencies = np.zeros(k + 1)
        
        for i in range(number_of_blocks):
            block_data = binArr[m*i:m*(i+1)]
            max_run_count = 0
            run_count = 0

            # This will count the number of ones in the block
            for bit in block_data:
                if bit == 1:
                    run_count += 1
                else:
                    max_run_count = max(max_run_count, run_count)
                    run_count = 0
            # For the case of having ended with a 1
            max_run_count = max(max_run_count, run_count)
            if max_run_count <= v_values[0]:
                frequencies[0] += 1
            elif max_run_count >= v_values[-1]:
                frequencies[-1] += 1

            else:
                frequencies[max_run_count - v_values[0]] += 1 # +1 as 0 corresponds to v_Values[0]

        # Compute xObs
        xObs = np.sum((frequencies - (number_of_blocks * pi_values))**2 / (number_of_blocks * pi_values))
        if verbose:
            print('Run Test (Longest Run of Ones in a Block) DEBUG BEGIN:')
            print("\tLength of input:\t\t\t\t", length_of_binary_data)
            print("\tSize of each Block:\t\t\t\t", m)
            print('\tNumber of Block:\t\t\t\t', number_of_blocks)
            print("\tValue of K:\t\t\t\t\t\t", k)
            print('\tValue of PIs:\t\t\t\t\t', pi_values)
            print('\tFrequencies:\t\t\t\t\t', frequencies)
            print('\txObs:\t\t\t\t\t\t\t', xObs)
        return xObs,k

        