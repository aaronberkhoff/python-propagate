import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler

import numpy as np

from copy import copy, deepcopy

from python_propagate.utilities.load_spice import load_spice,unload_spice



class GatingNetwork(nn.Module):
    def __init__(self, features, num_experts, hidden_dim=64,device = "cuda"):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(features, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.device = device
        self.num_experts = num_experts
        self.features = features
        # self.rnn = nn.GRU(features * num_experts, hidden_dim, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, num_experts)

        self.fc = nn.Sequential(
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(p=0.2)

        # self.conv = nn.Conv1d(in_channels=features, out_channels=hidden_dim, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.fc = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        #simple gating
        logits = self.fc(x)    

        #RNN gating
        # out, _ = self.rnn(x)  
        # logits = self.fc(out) 
    

        #CNN gating
        
        # x: (batch_size=T, num_experts, features)
        # x = x.view(x.shape[0],self.num_experts,self.features)
        # x = x.permute(1, 2, 0)  # (num_experts, features, T)
        # x = self.conv(x)        # (num_experts, hidden_dim, T)
        # x = self.relu(x)
        # x = x.permute(2, 0, 1)  # (T, num_experts, hidden_dim)
        # x = self.fc(x)          # (T, num_experts, num_experts)
        # # logits = x.mean(dim=-1)  # Reduce if needed
        # logits = x.squeeze(-1)

        weights = F.softmax(logits, dim=1)
        return weights, logits

    def train_model(self, x_train, y_train, num_epochs):

        #TODO need to account for time history

        self.to(self.device)

        x_train = torch.as_tensor(x_train,dtype=torch.float32).to(self.device)
        y_train = torch.as_tensor(y_train,dtype=torch.float32).to(self.device)

        #normalize data

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            gating_weights, gating_logits = self(x_train)
            # gating_weights = gating_weights.unsqueeze(-1)
            # x_train_reshaped = x_train.view(x_train.shape[0], self.num_experts, self.features)

            weighted_output = torch.sum(gating_weights * x_train, dim=1)  # Shape: (T,F)
            # weighted_output = torch.sum(gating_weights * x_train_reshaped, dim=1)  # Shape: (T,F)

            # Compute loss
            sigma = .01
            # loss = gaussian_nll(weighted_output, y_train, sigma)
            loss = mse_loss(weighted_output, y_train)
            loss += entropy_loss(gating_weights)
            # loss = cross_entropy(weighted_output, y_train, gating_logits= gating_logits)
            

            # Backpropagation
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


def gaussian_nll(y_pred, y_true, sigma):
    return ((y_pred - y_true) ** 2 / (2 * sigma**2)).mean()

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

def cross_entropy(y_pred, y_true,gating_logits):

    errors = torch.mean((y_pred - y_true.unsqueeze(1))**2, dim=-1)  # (T, N)
    true_expert = torch.argmin(errors, dim=1)  # (T,)

    loss = F.cross_entropy(gating_logits, true_expert)

    return loss

def entropy_loss(gating_weights):
    # gating_weights: shape (T, N)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-8
    entropy = -torch.sum(gating_weights * torch.log(gating_weights + epsilon), dim=-1)  # shape: (T,)
    return entropy.mean()





class HME:

    def __init__(self, experts, features = 6, device = 'cuda'):
        self.experts = experts
        self.gating_network = GatingNetwork(features=features, num_experts=len(experts),device=device)
        self.pos_scaler = MinMaxScaler()
        self.vel_scaler = MinMaxScaler()
        self.device = device




    def run(self,observations,num_epochs = 100, parallel = 0):

        self.process_experts(parallel=parallel)

        #extract data
        #TODO consider

        y_train = observations[:,1:] 

        y_train[:,:3] = self.pos_scaler.fit_transform(observations[:,:3])
        y_train[:,3:] = self.vel_scaler.fit_transform(y_train[:,3:])

        all_expert_states = [np.asarray(np.array([state.compile() for state in expert.state_data])) for expert in self.experts]
        all_expert_states = [np.hstack((self.pos_scaler.transform(expert[:,:3]),self.vel_scaler.transform(expert[:,3:]))) for expert in all_expert_states]
        
        expert_tensor = np.stack(all_expert_states, axis=1)  # shape: (T, N, F)


        T, N, F = expert_tensor.shape

        # Flatten last two dimensions: (T, N*F)
        # x_train = expert_tensor.reshape(T, N * F)
        x_train = expert_tensor

        self.gating_network.train_model(x_train, y_train, num_epochs=num_epochs)

        self.gating_network.eval()  # Set to evaluation mode

        with torch.no_grad():
            x_train = torch.as_tensor(x_train,dtype=torch.float32).to(self.device)
            gating_weights, gating_logits = self.gating_network(x_train)


        weights_np = gating_weights.squeeze().cpu().numpy()

        return weights_np

    def _process_expert(self,expert):
        #TODO need to account for time history
        load_spice() # TODO need to handle this if complex srp is a dynamics
        local_expert = deepcopy(expert)
    
        local_expert.propagate()

        return local_expert
    
    def process_experts(self,parallel = 0):

        

        if parallel:
            import concurrent.futures
            updated_experts = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = list([executor.submit(self._process_expert,expert) for expert in self.experts])
                for future in concurrent.futures.as_completed(futures):
                    expert = future.result()
                    updated_experts.append(expert)
            
            self.experts = updated_experts

        else:

            for expert in self.experts:

                expert.propagate()

        

            









