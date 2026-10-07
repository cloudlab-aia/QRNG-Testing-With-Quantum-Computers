# Improvements made by Lucas Hernandez Bellon (2026)
import numpy as np
from scipy.special import ndtr
class CumulativeSums:

    @staticmethod
    def cumulative_sums_test(binary_data:str, mode=0, verbose=False):
        """
        from the NIST documentation http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf

        The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative sum of
        adjusted (-1, +1) digits in the sequence. The purpose of the test is to determine whether the cumulative sum of
        the partial sequences occurring in the tested sequence is too large or too small relative to the expected
        behavior of that cumulative sum for random sequences. This cumulative sum may be considered as a random walk.
        For a random sequence, the excursions of the random walk should be near zero. For certain types of non-random
        sequences, the excursions of this random walk from zero will be large.

        :param      binary_data:    a binary string
        :param      mode            A switch for applying the test either forward through the input sequence (mode = 0)
                                    or backward through the sequence (mode = 1).
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)

        """
        
        # Convert binary_Data to numpy array for optimization purposes
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        
        length_of_binary_data = len(binArr)


        # Determine whether forward or backward data
        if not mode == 0:
            binArr = binArr[::-1]
        
        # Random Walk
        # Creation of the 1D cumulative sums array --> Changed to numpy methods
        stepsArr = np.where(binArr < 1, -1,1)
        counts = np.cumsum(stepsArr)

        # Compute the test statistic z =max1≤k≤n|Sk|, where max1≤k≤n|Sk| is the largest of the
        # absolute values of the partial sums Sk.
        abs_max = max(np.max(counts),-np.min(counts))
        sqrtN = np.sqrt(length_of_binary_data)
        sup = int(np.floor(0.25 * np.floor(length_of_binary_data / abs_max - 1))) + 1 # Last index of both summations
        # Changed calculations of the Standard Normal Cumulative Probability Distribution Function to scipy function for vectorising purposes
        # Vectorized the for with an arange of integers
        k4 = 4*np.arange(int(np.floor(-0.25*length_of_binary_data / abs_max + 1)),sup,1)
        terms_one = ndtr((k4 + 1)*abs_max/sqrtN) - ndtr((k4-1)*abs_max/sqrtN)

        k4 = 4*np.arange(int(np.floor(0.25 * -length_of_binary_data / abs_max - 3)),sup,1)
        terms_two = ndtr((k4+3)*abs_max/sqrtN) - ndtr((k4+1)*abs_max/sqrtN)

        p_value = 1.0 - np.sum(terms_one) + np.sum(terms_two)

        if verbose:
            print('Cumulative Sums Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tMode:\t\t\t\t', mode)
            print('\tValue of z:\t\t\t', abs_max)
            print('\tP-Value:\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value >= 0.01))