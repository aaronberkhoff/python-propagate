import numpy as np
from python_propagate.filters.hme_gate_Jah_GaiaVerse import JahGaiaVerseGate

class GatingNetwork:
    def __init__(self, eta = 1.75):
        self.eta = eta
        pass
    
    # Filter Conditional Probability
    def filter_cond_prob(self, howManyObs, innov_covariance, range_residuals):
        np.set_printoptions(precision=15)
        
        # Initialize filter conditional probability
        f_za = np.zeros(howManyObs)

        for i in range(howManyObs):
            r = range_residuals[i]
            W = innov_covariance[i][:][:]
            W = np.mean(W)  # Assuming covariance is a matrix

            f_za[i] = 1 / np.sqrt(2 * np.pi * W) * np.exp(-(r ** 2) / (2 * W))
            
        return f_za

    # Gating Network
    def gating_net(self, f_za, z_values):
        np.set_printoptions(precision=15)

        num_experts = len(f_za)
        ai = np.ones(num_experts)  # Initialize ai for each expert

        eta = self.eta  # Learning-rate parameter
        ui = np.zeros(num_experts)
        eu = np.zeros(num_experts)
        gi = np.zeros(num_experts)
        fg = np.zeros(num_experts)
        ai_hist = np.zeros((len(f_za[0]), num_experts))  # Shape is (howManyObs, num_experts)

        eu_hist = np.zeros((len(ai), len(f_za[0])))
        gi_hist = np.zeros((len(ai), len(f_za[0])))
        fg_hist = np.zeros((len(ai), len(f_za[0])))



        for i in range(len(f_za[0])):  # Loop over all observations

            for j in range(len(ui)):
                ui[j] = z_values[j][i] * ai[j]


            for j in range(len(ui)):
                eu[j] = np.exp(ui[j])
                    
                
                euSum = np.sum(eu)

                # Gating Weights via softmax 
                for j in range(len(ui)):
                    gi[j] = eu[j]/euSum

                for j in range(len(ui)):
                    fg[j] = f_za[j][i] * gi[j]

                fgSum = np.sum(fg)


                # Posteriori Probability and Grsadient Learning Rule update
                for j in range(len(ui)):
                    if fg[j] == 0:
                        hi = 1E-10
                        ai[j] = ai[j]
                    else:
                        hi = fg[j]/fgSum
                        ai[j] = ai[j] + (eta*(hi - gi[j]))

                ai_hist[i,:] = ai

                eu_hist[:,i] = eu
                gi_hist[:,i] = gi
                fg_hist[:,i] = fg

        return gi, ai_hist

    # Gating Network
    def run(self, results, measurements, howManyObs):
        num_experts = len(results)
        residuals = [expert.residuals_data[0]  for expert in results]
        innov_covariances = [expert.measurement_covariance_hist for expert in results]
        mean = [expert.measurement_mean_hist for expert in results]
        xhat_list = [expert.state_estimate_hist for expert in results]
        mu_list = [expert.state_mean_hist for expert in results]
        estimated_covariance_list = [expert.covariance_estimate_hist for expert in results]

        # Create a list of filter conditional probabilities (f_za)
        f_za = [self.filter_cond_prob(howManyObs, innov_covariances[i], residuals[i]) for i in range(num_experts)]


        # Jah Weighting
        weighting = JahGaiaVerseGate()
        # f_za = [weighting.gate(xhat_list, estimated_covariance_list, mu_list) for i in range(num_experts)]
        f_za = [weighting.gate(residuals[i], estimated_covariance_list[i]) for i in range(num_experts)]


        # Get neuron weights 
        gating_results = self.gating_net(f_za, residuals)

        ai_hist = gating_results[1]

        # Apply softmax column-wise
        ai_hist_exp = np.exp(ai_hist)  # Exponentiate each element in the array
        ai_hist_sum = np.sum(ai_hist_exp, axis=1, keepdims=True)  # Sum along each row
        ai_hist = ai_hist_exp / ai_hist_sum  # Normalize by row sum



        # Get the expert choices (from the last row of ai_hist)
        ai_list = ai_hist[-1, :]  # Last row gives the expert activation values
        expertChoice = np.argmax(ai_list)  # Select the expert with the max activation

        return expertChoice, ai_hist



