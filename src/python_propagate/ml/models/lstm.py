import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Gaussian weight uncertainty.
    """
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        # Mean and log variance for weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).fill_(-3))
        self.prior_std = prior_std
        self.out_features = out_features

    def forward(self, x):
        # Sample weights and biases using reparameterization
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)

    def kl_loss(self):
        # KL divergence between posterior and standard normal prior
        kl = 0.5 * (self.weight_mu.pow(2) + self.weight_logvar.exp() - self.weight_logvar - 1).sum()
        kl += 0.5 * (self.bias_mu.pow(2) + self.bias_logvar.exp() - self.bias_logvar - 1).sum()
        return kl / self.prior_std**2

class BayesianLSTMCell(nn.Module):
    """
    Bayesian LSTM Cell (single time step).
    """
    def __init__(self, input_size, hidden_size, prior_std=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prior_std = prior_std
        # Four gates: input, forget, cell, output
        self.x2h = BayesianLinear(input_size, 4 * hidden_size, prior_std)
        self.h2h = BayesianLinear(hidden_size, 4 * hidden_size, prior_std)
        

    def forward(self, x, hx):
        h, c = hx
        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, 2)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def kl_loss(self):
        return self.x2h.kl_loss() + self.h2h.kl_loss()
    


class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM for sequence processing.
    """
    def __init__(self, input_size, hidden_size, output_size, prior_std=0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = BayesianLSTMCell(input_size, hidden_size, prior_std)
        self.fc = BayesianLinear(hidden_size, output_size, prior_std)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
        out = self.fc(h)
        out = torch.tanh(out)
        return out
    
    def predict(self,x):

        return self.forward(x)
    
    def train_model(self, train_loader, num_epochs=10, lr=1e-3, beta=None, criterion=None, device='cpu'):
        self.to(device)
        # print(next(self.parameters()).device)
        if criterion is None:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if beta is None:
            beta = 1.0 / len(train_loader)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = self(x_batch)
                data_loss = criterion(output, y_batch[:, -1, :])
                kl = self.kl_loss()
                loss = data_loss + beta * kl
                # loss = data_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")    

    def kl_loss(self):
        return self.cell.kl_loss() + self.fc.kl_loss()


class LSTM(nn.Module):

    def __init__(self,input_size, hidden_size, output_size,device = 'cuda'):
        super().__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        self.activation = nn.Tanh()
        self.device = device

        

    def forward(self, x):

        out, (h,c) = self.lstm(x)
        
        out = self.fc(out[:,-1,:])

        out = self.activation(out)

        return out.unsqueeze(1)
    
    def train_model(self, train_loader, n_epochs, lr = .001):

        self.to(self.device)

        #define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(),lr=lr)

        for epoch in range(n_epochs):

            for x, y in train_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = self(x)

                loss = energy_loss(output) + criterion(output,y)
                # loss = criterion(output,y)

                loss.backward()
                optimizer.step()

            print(f'Epoch: {epoch + 1} Loss: {100 * loss.item():.4f}')

    def predict(self,x_test,device = 'cpu'):

        self.to(device=device)
        
        output = self(x_test)


        return output
    
    def predict_sequential(self, x_init, timesteps=1, window = 100, device='cuda'):
        self.to(device)
        self.eval()  # Disable dropout, etc.
        
        x_seq = x_init.to(device)
        preds = []

        for _ in range(timesteps):
            with torch.no_grad():
                x_next = self(x_seq[:, -window:, :])  # Only feed the last time step
            preds.append(x_next[:, 0, :].cpu().numpy())
            x_seq = torch.cat((x_seq, x_next), dim=1)

        return np.array(preds)

def energy_loss(predictions, mu = 1, threshold = -.5, weight = 1): 

    radii = torch.linalg.norm(predictions[:, :3], dim = 1) 
    velos = torch.linalg.norm(predictions[:, 3:], dim=1) 

    energy = 0.5 * torch.square(velos) - mu / radii

    energy_violation = torch.nn.functional.relu(energy - threshold)
    energy_penalty = torch.mean(energy_violation)

    return energy_penalty * weight
