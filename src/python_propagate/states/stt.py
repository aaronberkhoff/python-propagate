import numpy as np
import scipy as sp


class STT:
    """
    Class to represent the STT (State Transition Table) for a given agent.
    The STT is used to store the state transition probabilities between different states of the agent.
    """

    def __init__(self, state_mean, time, order, mu):

        self.n_states = max(state_mean.shape)
        self.order = order
        self.state_mean = state_mean
        self.mu = mu
        
    
        self._stts = []
        sma = self.state_mean[0]
        temp = 1 
        
        for m in range(1,order+1):
            temp *=  -(2*m + 1)/(2) 
            mean_anom = temp * np.sqrt(mu / sma ** (2*m + 3)) * time

            if m == 1:
                stt = np.eye(self.n_states)
                stt[5,0] = mean_anom
                self._stts.append(stt)

            else:
                stt = np.zeros(((m+1) * (self.n_states,)))
                index = (5,*(m * (0,)))
                stt[*index] = mean_anom
                self._stts.append(stt)

            

    def __call__(self, order):

        return self._stts[order-1]

    def propagate(self,state,order):
        final_deviation = np.zeros(self.n_states)  # Initialize final deviation
        stt_indices = 'ja'
        state_indices = 'a'

        for m  in range(1,order+1):
            
            factor = 1 / sp.special.factorial(m)

            index_str = f"{stt_indices},{state_indices} -> j"

            inputs = [self(m)] + [state] * (m)

            final_deviation += factor * np.einsum(index_str, *inputs)

            state_indices += ',' + chr(97+m)
            stt_indices += chr(97 + m)

            pass

        return final_deviation
