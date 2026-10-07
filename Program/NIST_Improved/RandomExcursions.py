# Improvements made by Lucas Hernandez Bellon (2026)
import numpy as np
from scipy.special import erfc as erfc
from scipy.special import gammaincc as gammaincc

class RandomExcursions:

    @staticmethod
    def get_pi_value(k, x):
        """
        This method is used by the random_excursions method to get expected probabilities. 
        Obtained from NIST 800-22 Rev.1a Section 3.14
        """
        out = np.zeros((len(k),len(x)))
        k0 = np.where(k == 0)[0]
        k4 = np.where((k >= 1) & (k < 5))[0]
        k5 = np.where(k > 4)[0]        
        
        out[k0,:] = 1 - 1.0 / (2 * abs(x))
        out[k4,:] = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * abs(x))) ** (k[k4,np.newaxis] - 1)
        out[k5,:] = (1.0 / (2 * abs(x))) * (1 - 1.0 / (2 * abs(x))) ** 4
        return out

    @staticmethod
    def random_excursions_test(binary_data:str, verbose=False, state=1):
        """
        from the NIST documentation http://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

        The focus of this test is the total number of times that a particular state is visited (i.e., occurs) in a
        cumulative sum random walk. The purpose of this test is to detect deviations from the expected number
        of visits to various states in the random walk. This test is actually a series of eighteen tests (and
        conclusions), one test and conclusion for each of the states: -9, -8, …, -1 and +1, +2, …, +9.

        :param      binary_data:    a binary string
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)
        """

        length_of_binary_data = len(binary_data)
        binArr = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        # Form the normalized (-1, +1) sequence X in which the zeros and ones of the input sequence (ε)
        # are converted to values of –1 and +1 via X = X1, X2, … , Xn, where Xi = 2εi – 1.
        sequence_x = np.where(binArr < 1, -1,1)

        # Compute partial sums Si of successively larger subsequences, each starting with x1. Form the set S
        cumulative_sum = np.zeros(length_of_binary_data + 2, dtype = np.int64) # We create S' directly instead of appending after computing cumsum
        cumulative_sum[1:-1] = np.cumsum(sequence_x)
        cropped_sum = cumulative_sum[abs(cumulative_sum) <= 4] + 4 # We only use summations with distances less than 5 from the center.
                                                                   # We also to positive values --> -4 ,4 to 0, 8. That way bincount can count
        # These are the states we are going to look at
        x_values = np.array([-4, -3, -2, -1,1, 2, 3, 4])
        nStates = len(x_values)
        # Identify all the locations where the cumulative sum visits each state. We now use bincounts
        J = np.where(cropped_sum == 4)[0] # Index of each 0 (for marking cycles). Remember that 0 now corresponds to value 4 in cropped sum
        # cycle 0: (J[1],J[0]), cycle 1: (J[2],J[1]), etc. There is always 1 cycle 
        nCycles = len(J)-1 # As they come in pairs

        state_count = np.zeros((nStates+1,nCycles), dtype = np.uint32)
        for j in range(nCycles): # We count the visits of each state in each cycle with a cropped bincount
            state_count[:,j] = np.bincount(cropped_sum[J[j]:J[j+1]] , minlength = nStates+1) # 0s are present (the leftmost one of the interval) 
                                                                                             # so that the counts are 0,1,2,3,4,5,6,7,8
        state_count = np.delete(state_count,4, axis = 0) # thus, we removed index 4 as it corresponds to the index of 0                                                                                
        
        # Su was an errata, I think they wanted to use nu, as nu_k(x) is the total number of cycles in which state x occurs exactly k times
        kArr = np.arange(0,6,1)
        clippedArr = np.minimum(state_count, 5) # As k = 5 must have also values # > 5 (nStates,nCycles)
        nu = np.stack([np.sum(clippedArr == k,axis = 1) for k in kArr]) # (nk = 6,nStates)

        piArr = RandomExcursions.get_pi_value(kArr, x_values)
        inner_term = nCycles * piArr # (nk, nStates)
        xObs = np.sum(1.0 * (nu - inner_term) ** 2 / inner_term, axis = 0)
        p_values = gammaincc(2.5, xObs / 2.0)

        result = np.zeros((nStates,5),dtype = object)
        names = ['-4','-3','-2','-1','+1','+2','+3','+4'] # Could be removed, but we leave the return as it is for compatibility with the library before the optimization
        result[:,0] = names
        result[:,1] = x_values
        result[:,2] = xObs
        result[:,3] = p_values
        result[:,4] = p_values >= 0.01

        if verbose:
            print('Random Excursion Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\t\t STATE \t\t\t xObs \t\t\t\t\t\t p_value  \t\t\t\t\t Result')
            for i in range(nStates):
                print('\t\t', x_values[i], ' \t\t ', xObs[i],' \t\t ', p_values[i])
            print('DEBUG END.')
        return result

    @staticmethod
    def variant_test(binary_data:str, verbose=False):
        """
        from the NIST documentation http://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

        :param binary_data:
        :param verbose:
        :return:
        """
        length_of_binary_data = len(binary_data)
        int_data = np.frombuffer(binary_data.encode('ascii'),np.uint8) - ord('0') # We extract the last bit of ascii codification of '0' and '1'
        sum_int = np.where(int_data < 1, -1,1)
        cumulative_sum = np.zeros(length_of_binary_data + 2, dtype = np.int32) # We add the first and last 0 as NIST advises
        cumulative_sum[1:-1] = np.cumsum(sum_int)
        
        states = np.arange(-9,10,1)
        nStates = len(states)
        # Counting how many times each particular state is visited
        countArr = np.bincount(cumulative_sum[abs(cumulative_sum) <= 9] + 9,minlength = nStates) # Bincount only counts with posive values, so we transform -9, ... 9 to 0, ... 18
        
        li_data = countArr[states != 0] # Position of 0 in states corresponds to the same after the bincount
        J = countArr[9] -1 # Number of cycles (number of steps between returning to 0 in the cusum). Index 9 is the one with 0
        states = states[states != 0] # We pop out the 0 state
        nStates -= 1 # We removed 0 state
        # Now that it was counted, we compute the associated pvalues
        p_values = erfc(abs(li_data-J)/np.sqrt(2*J*(4*abs(states)-2)))

        if verbose:
            print('Random Excursion Variant Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tValue of j:\t\t', J)
            print('\tP-Values:')
            print('\t\t STATE \t\t COUNTS \t\t P-Value')
            for i in range(nStates):
                print('\t\t', states[i], li_data[i], p_values[i])
            print('DEBUG END.')
        
        names = ['-9','-8','-7','-6','-5','-4','-3','-2','-1','+1','+2','+3','+4','+5','+6','+7','+8','+9']
        result = np.zeros((nStates,5)) # NOTE: NOW EVEN STATES THAT WERENT REACHED HAVE PVALUES; BEFORE ONLY REACHED ONES WHERE OUTPUTTED
        result[:,0] = names
        result[:,1] = states
        result[:,2] = li_data
        result[:,3] = p_values
        result[:,4] = p_values >= 0.01
        return result

