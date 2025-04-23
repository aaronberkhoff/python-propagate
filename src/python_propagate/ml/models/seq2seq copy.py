import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
import time

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




class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(output_size, 2 * hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        out = self.activation(self.fc(out))
        return out, h, c



class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cuda'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers=2, dropout=0.5)
        self.decoder = Decoder(output_size, hidden_size, num_layers=2, dropout=0.0)
        self.device = device
        self.log_vars = nn.ParameterDict({
                        'mse': nn.Parameter(torch.zeros(1)),
                        'energy1': nn.Parameter(torch.zeros(1)),
                        'energy2': nn.Parameter(torch.zeros(1)),
                        'sym': nn.Parameter(torch.zeros(1)),
                        'pinn': nn.Parameter(torch.zeros(1)),
                        })

    def forward(self, src, tgt_len, teacher_forcing_ratio=0.1, tgt=None):
        batch_size = src.size(0)
        output_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, output_size).to(self.device)

        # Encode input
        h, c = self.encoder(src)

        # Start decoding with last value of src
        decoder_input = src[:, -1:, :output_size]  # (batch, 1, output_size)

        for t in range(tgt_len):
            out, h, c = self.decoder(decoder_input, h, c)
            outputs[:, t:t+1, :] = out

            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1, :]
            else:
                decoder_input = out  # use model prediction as next input

        return outputs
    
    def train_model(self, train_loader, n_epochs, scaler, lr=0.001):
        self.to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

        start_time = time.time()
        for epoch in range(n_epochs):
            total_mse = 0.0
            total_energy1 = 0.0
            total_energy2 = 0.0
            total_symplectic = 0.0
            total_pinn = 0.0
            total_batches = 0
            teacher_forcing_ratio = max(0.75 * (0.99 ** epoch), 0.01)

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = self(x, tgt_len=y.size(1), tgt=y, teacher_forcing_ratio=teacher_forcing_ratio)

                # Compute individual losses
                mse_loss = criterion(output, y)
                energy1 = energy_difference_loss(output, y, scaler, weight=0.0)
                energy2 = energy_loss(output, scaler, weight=1.0)
                sym_loss = symplectic_loss(output, y, scaler, dt=30, weight=0.0)
                pin_loss = pinn_loss(output, y, scaler, weight=2.0)

                # Dynamic Kendall weighting
                loss = (
                    torch.exp(-self.log_vars['mse']) * mse_loss + self.log_vars['mse'] +
                    torch.exp(-self.log_vars['energy1']) * energy1 + self.log_vars['energy1'] +
                    torch.exp(-self.log_vars['energy2']) * energy2 + self.log_vars['energy2'] +
                    torch.exp(-self.log_vars['sym']) * sym_loss + self.log_vars['sym'] +
                    torch.exp(-self.log_vars['pinn']) * pin_loss + self.log_vars['pinn']
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                optimizer.step()

                # Track unnormalized losses for logging
                total_mse += mse_loss.item()
                total_energy1 += energy1.item()
                total_energy2 += energy2.item()
                total_symplectic += sym_loss.item()
                total_pinn += pin_loss.item()

                total_batches += 1

            avg_mse = total_mse / total_batches
            avg_energy1 = total_energy1 / total_batches
            avg_energy2 = total_energy2 / total_batches
            avg_sym = total_symplectic / total_batches
            avg_pinn = total_pinn / total_batches

            avg_total = avg_mse + avg_energy1 + avg_energy2 + avg_sym + avg_pinn
            scheduler.step(avg_total)

            # print_training_status(epoch+1, avg_mse, avg_energy1, avg_energy2, avg_total, lr, teacher_forcing_ratio)
            msg = (f"Epoch {epoch+1}" + 
                   f"| MSE Loss: {avg_mse:.4e}" + 
                   f"| Sym: {avg_sym:.4e}" + 
                   f"| PINN: {avg_pinn:.4e}" + 
                   f"| Energy Loss: {avg_energy2:.4e}" +
                   f"| Energy Diff: {avg_energy1:.4e}" +
                   f"| Total Loss: {avg_total:.4e}" + 
                   f"| LR: {scheduler.get_last_lr()[0]:4e}" +
                   f"| TFR: {teacher_forcing_ratio:.2e}")

            print(msg, end="\r", flush=True)

            # print(f"Epoch {epoch+1} | MSE Loss: {avg_mse:.6f} | Energy Loss: {avg_energy:.6f} | Total Loss: {avg_total:.6f}")

                # print_training_status(epoch+1, avg_mse, avg_energy1, avg_energy2, avg_total, lr, teacher_forcing_ratio)
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\n" + msg)
        print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


    
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
            predictions = self(src_seq, tgt_len=tgt_len, teacher_forcing_ratio=0.0, tgt=None)
        
        return predictions    
    

def duration_to_steps(duration_seconds, dt=30):
    return int(duration_seconds / dt)


def energy_loss(predictions, scaler, mu=1, weight=1.0):
    """
    Penalize the difference in orbital energy between predictions and targets.

    Args:
        predictions: (batch, seq_len, 6) — 3 pos + 3 vel
        mu: gravitational parameter (default = 1.0)
        weight: scaling factor for loss

    Returns:
        Scalar energy difference loss
    """
    

    def specific_energy(x):

        pos = scaler.pos_bound * x[:,:,:3]
        vel = scaler.vel_bound * x[:,:,3:]

        r = torch.linalg.norm(pos, dim=-1)
        v = torch.linalg.norm(vel, dim=-1)
        return 0.5 * v**2 - mu / r

    pred_energy = specific_energy(predictions)
    
    energy_violation = torch.nn.functional.relu(pred_energy)
    return energy_violation.mean() * weight


def energy_difference_loss(predictions, targets, scaler, mu=1, weight=1.0):
    """
    Penalize the difference in orbital energy between predictions and targets.

    Args:
        predictions: (batch, seq_len, 6) — 3 pos + 3 vel
        targets: (batch, seq_len, 6)
        mu: gravitational parameter (default = 1.0)
        weight: scaling factor for loss

    Returns:
        Scalar energy difference loss
    """
    

    def specific_energy(x):

        pos = scaler.pos_bound * x[:,:,:3]
        vel = scaler.vel_bound * x[:,:,3:]

        r = torch.linalg.norm(pos, dim=-1)
        v = torch.linalg.norm(vel, dim=-1)
        return 0.5 * v**2 - mu / r

    pred_energy = specific_energy(predictions)
    true_energy = specific_energy(targets)

    energy_diff = pred_energy - true_energy
    energy_loss = torch.mean(torch.square(energy_diff))  # MSE on energy difference

    return energy_loss * weight

def symplectic_loss(predictions, targets, scaler, dt=30.0, mu=398600.4415, weight=1.0):
    """
    Penalize the difference between the predicted and true position/velocity updates,
    ensuring that the system respects symplecticity.

    Args:
        predictions: (batch, seq_len, 6) — 3 pos + 3 vel predicted values
        targets: (batch, seq_len, 6) — 3 pos + 3 vel true values
        scaler: Scaler object used to inverse transform the position and velocity
        dt: time step (default = 1.0)
        mu: gravitational parameter (default = 398600.4415)
        weight: scaling factor for loss

    Returns:
        Scalar symplectic loss
    """
    
    def compute_position_velocity_loss(pred, target, dt):
        # Extract positions and velocities from predictions and targets
        pred_pos = scaler.inverse_position(pred[:, :, :3])
        pred_vel = scaler.inverse_velocity(pred[:, :, 3:])
        target_pos = scaler.inverse_position(target[:, :, :3])
        target_vel = scaler.inverse_velocity(target[:, :, 3:])

        # Compute the position and velocity difference
        pos_diff = pred_pos - target_pos - dt * pred_vel
        vel_diff = pred_vel - target_vel - dt * (mu / torch.norm(target_pos, dim=-1, keepdim=True)**3 * target_pos)

        # Compute the loss for both position and velocity
        pos_loss = torch.mean(torch.square(pos_diff))
        vel_loss = torch.mean(torch.square(vel_diff))
        return pos_loss, vel_loss

    # Calculate the position and velocity losses
    pos_loss, vel_loss = compute_position_velocity_loss(predictions, targets, dt)

    # Return the total loss, scaled by the provided weight
    total_loss = pos_loss + vel_loss
    return total_loss * weight

def pinn_loss(predictions, targets, scaler,  mu=398600.4415, weight=1.0):
    """
    Penalize the difference between the predicted and true acceleration,
    ensuring that the system obeys the governing equations of motion.

    Args:
        predictions: (batch, seq_len, 6) — 3 pos + 3 vel predicted values
        targets: (batch, seq_len, 6) — 3 pos + 3 vel true values
        scaler: Scaler object used to inverse transform the position and velocity
        dt: time step (default = 1.0)
        mu: gravitational parameter (default = 398600.4415)
        weight: scaling factor for loss

    Returns:
        Scalar PINN loss
    """

    def compute_acceleration(pred, target, mu):
        # Extract positions and velocities
        pred_pos = scaler.inverse_position(pred[:, :, :3])
        # pred_vel = scaler.inverse_velocity(pred[:, :, 3:])
        target_pos = scaler.inverse_position(target[:, :, :3])
        # target_vel = scaler.inverse_velocity(target[:, :, 3:])

        # Compute the true and predicted accelerations
        true_acc = -mu / torch.norm(target_pos, dim=-1, keepdim=True)**3 * target_pos
        pred_acc = -mu / torch.norm(pred_pos, dim=-1, keepdim=True)**3 * pred_pos
        

        # Compute the acceleration loss (penalize the residual of acceleration)
        acc_loss = torch.mean(torch.square(pred_acc - true_acc))
        return acc_loss

    # Calculate the acceleration loss
    acc_loss = compute_acceleration(predictions, targets, mu)

    # Return the total loss, scaled by the provided weight
    return acc_loss * weight


def print_training_status(epoch, avg_mse, avg_energy1, avg_energy2, avg_total, lr, tfr):
    # Move cursor up by 7 lines (number of lines to overwrite)
    sys.stdout.write("\033[F" * 7)  # ANSI escape code
    sys.stdout.write("\r")  # Return to beginning of the line
    sys.stdout.flush()

    # print(f"\n")
    print(f"Epoch: {epoch}")
    print(f"MSE Loss: {avg_mse:.6f}")
    print(f"Energy Diff: {avg_energy1:.6f}")
    print(f"Energy Loss: {avg_energy2:.6f}")
    print(f"Total Loss: {avg_total:.6f}")
    print(f"LR: {lr:.4e}")
    print(f"TFR: {tfr:.2f}")