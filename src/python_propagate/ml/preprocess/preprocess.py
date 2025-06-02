import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler

class NonDimensional:
    def __init__(self, sma, mu=398600.4415):
        """
        Parameters:
        - sma: reference semi-major axis [km]
        - mu: gravitational parameter [km^3/s^2]
        - max_position: expected max |r| in units of SMA (e.g. 1.5 * SMA)
        - max_velocity: expected max |v| in units of canonical velocity (e.g. 1.2 * sqrt(mu/sma))
        """
        self.sma = sma
        self.mu = mu
        self.time_transform = 1 / np.sqrt(sma**3 / mu)
        self.position_transform = 1 / sma
        self.velocity_transform = 1 / np.sqrt(mu / sma)

        self.pos_bound = None
        self.vel_bound = None

    @property
    def energy_bound(self):
        
        return 0.5 * (self.vel_bound)**2 - 1 / self.pos_bound

    def fit_position(self, data: np.ndarray) -> np.ndarray:
        # Step 1: Canonical transform
        # normalized = data * self.position_transform

        # max_positions = np.max(np.linalg.norm(normalized,axis=1))
        self.pos_bound = max_positions

        # Step 2: Scale to (-1, 1)
        return normalized / max_positions

    def fit_velocity(self, data: np.ndarray) -> np.ndarray:
        # Step 1: Canonical transform
        normalized = data * self.velocity_transform

        max_velocity = np.max(np.linalg.norm(normalized,axis=1))
        self.vel_bound = max_velocity

        # Step 2: Scale to (-1, 1)
        return normalized / max_velocity

    def fit_time(self, data: np.ndarray) -> np.ndarray:
        return data * self.time_transform

    def inverse_position(self, norm_data: np.ndarray) -> np.ndarray:
        return norm_data * self.pos_bound / self.position_transform

    def inverse_velocity(self, norm_data: np.ndarray) -> np.ndarray:
        return norm_data * self.vel_bound / self.velocity_transform
    
class MinMax:
    def __init__(self, feature_range = (-1,1), mu=398600.4415):
        """
        Parameters:
        - sma: reference semi-major axis [km]
        - mu: gravitational parameter [km^3/s^2]
        - max_position: expected max |r| in units of SMA (e.g. 1.5 * SMA)
        - max_velocity: expected max |v| in units of canonical velocity (e.g. 1.2 * sqrt(mu/sma))
        """
        self.pos_scaler = MinMaxScaler(feature_range=feature_range)
        self.vel_scaler = MinMaxScaler(feature_range=feature_range)
        self.mu = mu


    def fit_position(self, data: np.ndarray) -> np.ndarray:

        return self.pos_scaler.fit_transform(data)

    def fit_velocity(self, data: np.ndarray) -> np.ndarray:
        
        return self.vel_scaler.fit_transform(data)
    

    def inverse_position(self, norm_data: np.ndarray) -> np.ndarray:
        return self.pos_scaler.inverse_transform(norm_data)

    def inverse_velocity(self, norm_data: np.ndarray) -> np.ndarray:
        return self.vel_scaler.inverse_transform(norm_data)

class Seq2SeqDataModule:
    class Seq2SeqDataset(Dataset):
        def __init__(self, data, input_window, target_window, overlap):
            self.x, self.y = self.create_batches(data, input_window, target_window, overlap)

        def create_batches(self, data, input_window, target_window, overlap):
            step_size = int(input_window * (1 - overlap))
            if step_size < 1:
                raise ValueError("Overlap too high, step_size < 1")

            x_batches = []
            y_batches = []

            for start in range(0, len(data) - input_window - target_window + 1, step_size):
                x = data[start : start + input_window]
                y = data[start + input_window : start + input_window + target_window]
                x_batches.append(x)
                y_batches.append(y)

            return np.array(x_batches), np.array(y_batches)


        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

    def __init__(self, data, input_window, target_window, overlap=0.0, batch_size=32, shuffle=True):
        self.data = data
        self.input_window = input_window
        self.target_window = target_window
        self.overlap = overlap
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        dataset = self.Seq2SeqDataset(
            self.data,
            self.input_window,
            self.target_window,
            self.overlap
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,pin_memory=True)

