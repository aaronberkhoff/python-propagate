import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import sys
import time
import random

from python_propagate.ml.models.lstm import BayesianLinear, BayesianLSTMCell

MSE = nn.MSELoss()

# torch.random.seed(100)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)               # For current GPU
    torch.cuda.manual_seed_all(seed)            # If using multi-GPU
    np.random.seed(seed)                        # Numpy seed
    random.seed(seed)                           # Python seed
    torch.backends.cudnn.deterministic = True   # Make CUDA deterministic (slow but reproducible)
    torch.backends.cudnn.benchmark = False 

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layernorm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        outputs = self.layernorm(outputs)
        # h and c: shape (2 * num_layers, batch, hidden_size)

        # Reshape bidirectional hidden states into (num_layers, batch, 2 * hidden_size)
        h = h.view(self.num_layers, 2, x.size(0), self.hidden_size)  # (num_layers, 2, batch, hidden)
        c = c.view(self.num_layers, 2, x.size(0), self.hidden_size)

        # Concatenate forward and backward hidden states
        h_cat = torch.cat((h[:, 0], h[:, 1]), dim=-1)  # (num_layers, batch, 2 * hidden)
        c_cat = torch.cat((c[:, 0], c[:, 1]), dim=-1)

        return h_cat, c_cat
    
    def kl_loss(self):

        return 0.0
    

class BayesianEncoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, prior_std=0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.bayesian_lstm = BayesianLSTMCell(input_size, hidden_size, prior_std)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            h, c = self.bayesian_lstm(x[:, t], (h, c))

        return h, c

    def kl_loss(self):
        return self.bayesian_lstm.kl_loss()



class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(output_size, 2 * hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        out = self.activation(self.fc(out))
        # out = self.fc(out)
        return out, h, c
    
class BayesianDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prior_std=1.0, direction = 1):
        super().__init__()
        # self.lstm = BayesianLSTMCell(input_size, hidden_size, prior_std)
        
        # self.fc = BayesianLinear(hidden_size, output_size, prior_std)

        #Bidirectional Encoder
        self.lstm = BayesianLSTMCell(input_size, hidden_size, prior_std, direction = direction)
        self.fc = BayesianLinear(direction * hidden_size, output_size, prior_std)
        # self.activation = nn.Tanh()

        # Add projection layers to reduce from bidirectional encoder hidden size
        # self.h_proj = nn.Linear(2 * hidden_size, hidden_size)
        # self.c_proj = nn.Linear(2 * hidden_size, hidden_size)


    def forward(self, x, h, c):
        # Project from 2*hidden → hidden
        #Used for bidirectional encoder
        # h = self.h_proj(h)  # Project hidden state from 2*hidden to hidden
        # c = self.c_proj(c)  # Project cell state from 2*hidden to hidden
        
        # Pass through LSTM (ensure it's in the correct shape)
        h, c = self.lstm(x.squeeze(1), (h.squeeze(0), c.squeeze(0)))
        
        # Flatten the output from (batch_size, seq_len, hidden_size) to (batch_size * seq_len, hidden_size)
        # h_flat = h.view(-1, h.size(-1))  # Flatten to (batch_size * seq_len, hidden_size)
        
        # Apply Bayesian Linear layer

        output, var = self.fc(h)
        
        # Reshape output to match (batch_size, seq_len, output_size)
        # output = output.view(h.size(0), -1, output.size(-1))  # Reshape back to (batch_size, seq_len, output_size)

        # mean, logvar = torch.chunk(output, 2, dim=-1)  

        # output = self.activation(output)
        
        return output.unsqueeze(1), h, c, var.unsqueeze(1)

    def kl_loss(self):
        return self.lstm.kl_loss() + self.fc.kl_loss()




class Seq2SeqBayesian(nn.Module):
    set_seed(100)
    def __init__(self, input_size, hidden_size, output_size, device='cuda'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers=1, dropout=0.2)
        # self.decoder = Decoder(output_size, hidden_size, num_layers=2, dropout=0.2)

        # self.encoder = BayesianEncoder(input_size, hidden_size, prior_std=.01)
        self.decoder = BayesianDecoder(input_size, hidden_size, output_size, prior_std=0.01, direction=2)

        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, src, tgt_len, teacher_forcing_ratio=0.1, tgt=None):
        batch_size = src.size(0)
        output_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, output_size).to(self.device)
        vars = torch.zeros(batch_size, tgt_len, output_size).to(self.device)

        # Encode input
        h, c = self.encoder(src)

        # Start decoding with last value of src
        decoder_input = src[:, -1:, :output_size]  # (batch, 1, output_size)

        for t in range(tgt_len):
            out, h, c, var = self.decoder(decoder_input, h, c)
            outputs[:, t:t+1, :] = out
            vars[:, t:t+1, :] = var

            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1, :]
            else:
                decoder_input = out  # use model prediction as next input

        return outputs, vars
    
    def train_model(self, train_loader, n_epochs, scaler, lr=0.01):
        self.to(self.device)
        writer = SummaryWriter(log_dir="tests/ml/results/runs/bayesian")

        # Loss and optimizer
        

        # Collect all parameters of the model except log_vars
        model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        

        optimizer = optim.Adam(model_params, lr=lr)
        # optimizer = torch.optim.SGD(model_params, lr=0.1, momentum=0.9)
        # optimizer = torch.optim.RAdam(model_params, lr=lr)


        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=30, cooldown=30)

        start_time = time.time()
        for epoch in range(n_epochs):
            total_mse = 0.0
            total_energy1 = 0.0
            total_energy2 = 0.0
            total_symplectic = 0.0
            total_pinn = 0.0
            total_kl = 0.0
            total_mom = 0.0
            total_mahal = 0.0
            total_batches = 0
            teacher_forcing_ratio = max(0.75 * (0.999 ** epoch), 0.01)
            # teacher_forcing_ratio = .01
            kl_loss = 0
             

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                

                # Forward pass
                output, vars = self(x, tgt_len=y.size(1), tgt=y, teacher_forcing_ratio=teacher_forcing_ratio)

                # weight_logvar = self.decoder.fc.weight_logvar

                #Baysian requirement:
                kl_weight = 0.0
                kl_loss = (self.decoder.kl_loss()) * kl_weight
                kl_loss = torch.clamp(kl_loss,0,1)

                #mahal loss
            
                mahal_loss = mahalanobis_loss(output,vars,y,weight = 1.0,constant=10)
                mahal_loss = torch.clamp(mahal_loss,0,1)
                
                # Compute individual losses
                mse_loss = self.mse(output, y) * 1.0
                # mse_loss = torch.clamp(mse_loss,min=0,max=1.0)

                energy2 = energy_loss(output, scaler, weight=1.0)
                # energy2 = torch.clamp(energy2,min=0,max=1.0)
                sym_loss = approximate_symplectic_loss(output,scaler=scaler, weight=1.0,dt = 60.0)

                total_loss = (
                    mse_loss +
                    # energy1 +
                    energy2 +
                    # mom_loss +
                    sym_loss +
                    # pin_loss +
                    kl_loss +
                    mahal_loss
                )

                total_loss.backward()

                # Optional: gradient clipping to improve stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()


                # Track unnormalized losses for logging
                total_mse += mse_loss.item()
                total_energy2 += energy2.item()
                total_symplectic += sym_loss.item()
                total_kl += kl_loss.item()
                total_mahal += mahal_loss.item()

                total_batches += 1

            avg_mse = total_mse / total_batches
            avg_energy2 = total_energy2 / total_batches
            avg_sym = total_symplectic / total_batches
            avg_kl = total_kl / total_batches
            avg_mahal = total_mahal / total_batches

            avg_total = avg_mse + avg_energy2 + avg_sym + avg_kl + avg_mahal
            

            scheduler.step(total_loss.item())

            
            # sys.stdout.write("\033[F" "\r")
            # sys.stdout.flush()

            # Print nicely formatted summary of all losses
            print(
                f"Epoch {epoch+1:03d} | "
                f"MSE: {avg_mse:.3e} | "
                # f"Energy Diff: {avg_energy1:.3e} | "
                f"Energy: {avg_energy2:.3e} | "
                # f"Mom: {avg_mom:.3e} | "
                f"MAHAL: {avg_mahal:.3e} | "
                f"SYM: {avg_sym:.3e} | "
                f"KL: {avg_kl:.3e} | "
                f"Total: {avg_total:.3e} | "
                f"LR: {scheduler.optimizer.param_groups[0]['lr']:.1e} | "
                f"TFR: {teacher_forcing_ratio:.2f}"
            )
           
            
            writer.add_scalar("Loss/SYM", avg_sym, epoch)

            writer.add_scalar("Loss/Energy_Bound", avg_energy2, epoch)

            writer.add_scalar("Loss/MSE", avg_mse, epoch)

            writer.add_scalar("Loss/KL", avg_kl, epoch)
            writer.add_scalar("Loss/NLL", avg_mahal, epoch)

            
            with torch.no_grad():
                pred_flat = output.detach().cpu().reshape(-1, output.size(-1))
                target_flat = y.detach().cpu().reshape(-1, y.size(-1))
                diff = (pred_flat - target_flat)

                mae = torch.mean(torch.abs(diff)).item()
                rmse = torch.sqrt(torch.mean(diff ** 2)).item()

                writer.add_scalar("Error/MAE", mae, epoch)
                writer.add_scalar("Error/RMSE", rmse, epoch)

                # Optional: Log histogram of prediction errors
                writer.add_histogram("Error/Prediction_Diff", diff, epoch)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\nTraining completed in {:.2f} seconds ({:.2f} minutes)".format(elapsed_time, elapsed_time/60))



    
    def predict_variable_horizon(self, src_seq, duration_seconds, dt_seconds=30):
        """
        Predict a variable-length output sequence given the source sequence.
        
        Args:
            self: trained Seq2Seq model
            src_seq (torch.Tensor): (batch, input_len, features)
            duration_seconds (int): how far ahead to predict
            dt_seconds (int): data sampling period in seconds (default: 30)
        
        Returns:
            torch.Tensor: (batch, target_len, features)
        """
        self.eval()
        tgt_len = int(duration_seconds / dt_seconds)

        src_seq = src_seq.to(self.device)
        
        with torch.no_grad():
            predictions, var = self(src_seq, tgt_len=tgt_len, teacher_forcing_ratio=0.0, tgt=None)
        
        return predictions, var    
    

def duration_to_steps(duration_seconds, dt=30):
    return int(duration_seconds / dt)


def energy_loss(predictions, scaler, weight=1.0):
    """
    Penalize the difference in orbital energy between predictions and targets.

    Args:
        predictions: (batch, seq_len, 6) — 3 pos + 3 vel
        mu: gravitational parameter (default = 1.0)
        weight: scaling factor for loss

    Returns:
        Scalar energy difference loss
    """
    

    def specific_energy(data):
        # Ensure data is a tensor
        if not torch.is_tensor(data):
            raise TypeError("Expected data to be a tensor of shape (batch, seq, features)")

        device = data.device
        dtype = data.dtype
        batch_size, seq_len, feat_dim = data.shape

        # Split into position and velocity
        pos_raw = data[:, :, :3].reshape(-1, 3).detach().cpu().numpy()  # (batch * seq, 3)
        vel_raw = data[:, :, 3:].reshape(-1, 3).detach().cpu().numpy()  # (batch * seq, 3)

        # Apply inverse scaling
        pos_inv = scaler.inverse_position(pos_raw).reshape(batch_size, seq_len, 3)
        vel_inv = scaler.inverse_velocity(vel_raw).reshape(batch_size, seq_len, 3)

        # Convert back to torch on the correct device and dtype
        pos = torch.tensor(pos_inv, device=device, dtype=dtype)
        vel = torch.tensor(vel_inv, device=device, dtype=dtype)

        # Compute norms
        r = torch.linalg.norm(pos, dim=-1)
        v = torch.linalg.norm(vel, dim=-1)

        # Specific mechanical energy
        return 0.5 * v**2 - scaler.mu / r

    pred_energy = specific_energy(predictions)
    
    energy_violation = torch.nn.functional.relu(pred_energy)
    return energy_violation.mean() * weight


def mahalanobis_loss(pred_mean, pred_logvar, target,weight = 1.0,constant = 1e-6):
    """
    Computes Mahalanobis loss between prediction and target.
    pred_mean: model predicted mean output
    pred_logvar: model predicted log variance
    target: ground truth
    """
    soft = nn.Softplus()
    
    diff = pred_mean - target         
    
           # Prediction error
    pred_logvar = torch.clamp(pred_logvar, min=-20, max=5) 

    loss = (pred_logvar / 2) + soft(diff.pow(2) / (torch.exp(pred_logvar))) + constant
    return loss.mean() * weight


def approximate_symplectic_loss(predictions, scaler, weight = 1.0, dt=60.0):
    """
    predictions: model predictions, shape (batch_size, seq_len, state_dim)
    scaler: your scaler to unnormalize back to real physical units
    dt: timestep between predictionss
    """

    # batch_size, seq_len, dim = predictions.shape

    def two_body_motion(q,p, scaler):

        r = torch.linalg.norm(q, dim=-1, keepdim=True)  # (batch, seq_len, 1)
        dpdt = -scaler.mu / r**3 * q  # (batch, seq_len, 3)
        dqdt = p

        return dqdt, dpdt

    device = predictions.device
    dtype = predictions.dtype
    batch_size, seq_len, _ = predictions.shape

    # Extract and reshape position data
    pred_pos = predictions[:, :, :3].reshape(-1, 3).detach().cpu().numpy()
    pred_vel = predictions[:, :, 3:].reshape(-1, 3).detach().cpu().numpy()

    # Apply inverse scaling
    pred_pos = scaler.inverse_position(pred_pos).reshape(batch_size, seq_len, 3)
    pred_vel = scaler.inverse_velocity(pred_vel).reshape(batch_size, seq_len, 3)

    # Convert back to torch tensors on the original device
    q = torch.tensor(pred_pos, device=device, dtype=dtype)
    p = torch.tensor(pred_vel, device=device, dtype=dtype)

    # Assume first half are positions (q), second half are momenta (p)
    # q = pred_pos
    # p = pred_vel

    

    # Compute finite differences
    dqdt = (q[:, 1:, :] - q[:, :-1, :]) / (dt * seq_len)
    dpdt = (p[:, 1:, :] - p[:, :-1, :]) / (dt * seq_len)

    # dqdt, dpdt = two_body_motion(q, p, scaler)


    # dq_dt = (-q[:, 2:, :] + 8*q[:, 1:-1, :] - 8*q[:, :-2, :] + q[:, :-3, :]) / (6 * dt * seq_len)
    # dp_dt = (-p[:, 2:, :] + 8*p[:, 1:-1, :] - 8*p[:, :-2, :] + p[:, :-3, :]) / (6 * dt * seq_len)

    # dq_dt = (q[:, 0:-1, :] - q[:, 1:, :]) / dt
    # dp_dt = (p[:, 0:-1, :] - p[:, 1:, :]) / dt

    # Hamiltonian consistency: q_dot and p_dot should have certain structure
    symplectic_residual = dqdt**2 + dpdt**2  # should be close to zero ideally

    # Loss: Mean squared symplectic residual
    # loss = torch.mean(torch.tanh(symplectic_residual**2))
    # loss = torch.mean(symplectic_residual)

    loss = MSE(torch.zeros_like(dqdt), dqdt)
    loss += MSE(torch.zeros_like(dpdt), dpdt)
    

    # loss = torch.clip(loss,min=0.0, max=1.0)
    # loss = torch.tanh(loss)

    return loss * weight


