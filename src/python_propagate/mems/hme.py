import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import random

import numpy as np

from copy import copy, deepcopy
import warnings

from datetime import datetime, timedelta

from python_propagate.utilities.load_spice import load_spice,unload_spice
from python_propagate.utilities.string_format import DATESTR
from python_propagate.utilities.units import DEG2RAD, RAD2DEG

from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.states import State
from python_propagate.mems.experts import Expert
from python_propagate.mems.loss import WeightedMSE, SurprisalLoss, EntropyLoss, NecessityLoss

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MaxNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-8):
        self.eps = eps  # to prevent division by zero
        self.max_vals_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.max_vals_ = np.max(np.abs(X), axis=0)
        self.max_vals_[self.max_vals_ < self.eps] = 1.0  # prevent divide by near-zero
        return self

    def transform(self, X):
        if self.max_vals_ is None:
            raise RuntimeError("You must fit the normalizer before calling transform.")
        return X / self.max_vals_

    def inverse_transform(self, X_norm):
        return X_norm * self.max_vals_
    

class LogNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, base=np.e):
        self.base = base
        self.fitted = False

    def fit(self, X):
        X = np.asarray(X)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("SignedLogScaler must be fit before transform.")
        X = np.asarray(X)
        return np.sign(X) * np.log1p(np.abs(X)) / np.log(self.base)

    def inverse_transform(self, X_log):
        if not self.fitted:
            raise RuntimeError("SignedLogScaler must be fit before inverse_transform.")
        return np.sign(X_log) * (np.exp(np.abs(X_log) * np.log(self.base)) - 1)

    




def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





class GatingNetwork(nn.Module):
    def __init__(self, features, num_experts, hidden_dim=64,device = "cuda"):
        super().__init__()
        
        # set_seed(147)
        self.device = device
        self.num_experts = num_experts
        self.features = features
        alpha = 1.0
        # self.loss = WeightedMSE(power=4,weight=alpha) + WeightedMSE(power=2,weight=(1-alpha))
        # self.loss = WeightedMSE(power=4) + WeightedMSE(power=2)
        # self.loss = EntropyLoss(power=4,weight=1.0) #+ EntropyLoss(power=2,weight=1.0)
        self.loss = EntropyLoss(power=2,weight=alpha) #+ EntropyLoss(power=2,weight=(1 - alpha))
        # self.loss =  WeightedMSE(power=4,weight=alpha)
        # self.loss = JahLoss()
        # self.loss = SurprisalLoss(power=2)
        

        # self.fc = nn.Sequential(
        #     nn.Linear(features, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 1),
        # )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_experts),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_experts),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

        kernel_size = 3
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(in_channels=hidden_dim,out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )


    def init_weights_equal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    def init_kaiming(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


    def forward(self, x):
        # #simple gating

        x = x.permute(0, 2, 1)
        x = self.conv(x)      # Shape: [T*N, 1]
        x = x.permute(0, 2, 1)

        logits = self.fc(x)
        temperature = 1.0
        # weights = torch.softmax(logits / temperature, dim=1)  # Softmax over experts for each timestamp
        weights = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)  # Softmax over experts for each timestamp

        #SUm of weights might not be one
        
        return weights, logits

    def train_model(self, residuals, num_epochs):

        self.to(self.device)
        self.train()

        # x_train = torch.as_tensor(x_train,dtype=torch.float32).to(self.device)
        x_train = torch.as_tensor(residuals,dtype=torch.float32).to(self.device)
        # y_train = torch.as_tensor(y_train,dtype=torch.float32).to(self.device)
        # y_train = y_train.unsqueeze(1)



        #normalize data
        lr = 1e-3
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        # optimizer = optim.SGD(self.parameters(),momentum=.9,lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,mode = 'min', factor=.9, patience=300, cooldown=100)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            gating_weights, gating_logits = self(x_train)
            
            # loss = weighted_mse_loss(x_train,weights=gating_weights)
            # loss = weighted_mse4_loss(x_train,weights=gating_weights)
            loss = self.loss(predicted = x_train,targets = torch.zeros_like(x_train), weights = gating_weights)

            # Backpropagation
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            # if scheduler.optimizer.param_groups[0]['lr'] < lr:

            #     print(f"Epoch {epoch}: Final Loss: {loss.item():.10e}")
            #     break


            if epoch % 100 == 0:
                # print(f"Epoch {epoch}: Total:{loss.item():.4f}, mse_loss = {ms_loss.item():.4f}, ent_loss = {en_loss.item():.4f}, mar_loss = {ma_loss.item():.4f}")
                print(f"Epoch {epoch}: Total:{loss.item():.10e}, LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")

            scheduler.step(loss)


class HME:

    def __init__(self, experts_file, device = 'cpu'):

        config = load_yaml(yaml_file=experts_file)
        try: 
            self.experts = config['experts']
            self.station = config['stations'][0]
            #TODO Make it capable with multiple stations
        except ValueError:
            print(f'Input file <{experts_file}> must have a experts field defined')

        
        
        # self.scaler = MinMaxScaler(feature_range=(-1,1))
        # self.scaler = MaxNormalizer()

        # self.scaler = RobustScaler()
        # self.scaler = LogNormalizer()
        self.scaler = StandardScaler()
        self.device = device
        self.features = []
        # self._valid_features = ['X_INERTIAL_KM','Y_INERTIAL_KM','Z_INERTIAL_KM',
        #                         'VX_INERTIAL_KMS','VY_INERTIAL_KMS','VZ_INERTIAL_KMS',
        #                         'LAT_DEG','LON_DEG','ALT_KM',
        #                         'RA_DEG', 'DEC_DEG',
        #                         'RANGE_KM', 'RANGE_RATE_KMS',
        #                         'AZ_DEG', 'EL_DEG', 
        #                         'FLUX_W_M2','FLUX_APPARENT_MAG']
        
        self._valid_features = ['X_INERTIAL_KM','Y_INERTIAL_KM','Z_INERTIAL_KM',
                                'VX_INERTIAL_KMS','VY_INERTIAL_KMS','VZ_INERTIAL_KMS',
                                ]
        
        

       


    def run(self,observations_dataframe,num_epochs = 1000, parallel = 0):

        times, observations, time_sec = self.process_observations(observations_dataframe)

        x_train, y_train, residuals = self.process_experts(times = times, observations=observations)

        gating_network = GatingNetwork(features=y_train.shape[-1], num_experts=len(self.experts),device=self.device)

        gating_network.train_model(residuals=residuals, num_epochs=num_epochs)

        gating_network.eval()  # Set to evaluation mode

        with torch.no_grad():
            x_train = torch.as_tensor(residuals,dtype=torch.float32).to(self.device)
            gating_weights, gating_logits = gating_network(x_train)


        weights_np = gating_weights.squeeze().cpu().numpy()

        expert_dict = {expert.name: {'epoch_time': None, 'probabilities': None} for expert in self.experts}

        for expert, weight in zip(self.experts, weights_np.T):

            expert.probability_data = weight
            expert_dict[expert.name]['epoch_time'] = times[1:]
            expert_dict[expert.name]['time_sec'] = time_sec[1:] / 3600
            expert_dict[expert.name]['probabilities'] = weight 

        return expert_dict   

    def process_observations(self,observation_dataframe):
        try:
            times = [time.decode('utf-8') for time in observation_dataframe['epoch_time'].values]
        except Exception:
            times = [str(time) for time in observation_dataframe['epoch_time'].values]
        pass

        observations = []

        for att in self._valid_features[:]:

            try:
                observations.append(observation_dataframe[att].values)
            except KeyError:
                print(f"Feature <{att}> not found in dataframe. Removing for analysis...")
                self._valid_features.remove(att)
                continue

        time_sec = observation_dataframe['time_sec']

        return times, np.array(observations).T, time_sec
    
    def process_experts(self,times, observations):

        #set the initial condition of the experts to the data:
        all_expert_states = [[] for _ in range(len(self.experts))]
        for i,(tk, tkm1, obs) in enumerate(zip(times[1:],times[0:-1],observations[0:-1,:6])):
            
            for j, expert in enumerate(self.experts):
                start_time = datetime.strptime(tkm1, DATESTR)
                end_time = datetime.strptime(tk, DATESTR)
                expert.state_data = []
                expert.state = State(position = obs[:3].astype(float), velocity = obs[3:6].astype(float), time = start_time)
                expert.propagate(duration = (end_time - start_time).total_seconds())
                final_state = expert.state_data[-1] 
                self.station.state.time = end_time
        
                state_data = list(final_state.compile())

                #calculate lat lon alt
                if 'LAT_DEG' and 'LON_DEG' and 'ALT_KM' in self._valid_features: 
                    state_data.extend([final_state.latlong[0]*RAD2DEG,final_state.latlong[1]*RAD2DEG,np.linalg.norm(final_state.position) - expert.scenario.central_body.radius])

                #expected measurements
                if 'RA_DEG' and 'DEC_DEG' in self._valid_features: 
                    ra, dec = self.station.calculate_ra_and_dec(state=final_state)

                    ra *= RAD2DEG 
                    ra = ra % 360

                    dec *= RAD2DEG
                    dec = max(-90, min(90,dec))
                    state_data.extend([ra,dec])

                if 'RANGE_KM' and 'RANGE_RATE_KMS' in self._valid_features: 
                    rho, rhodot = self.station.calculate_range_and_range_rate_from_target(state=final_state)
                    state_data.extend([rho,rhodot])

                if 'AZ_DEG' and 'EL_DEG' in self._valid_features: 
                    az, el = self.station.calculate_azimuth_and_elevation(state=final_state)
                    az *= RAD2DEG
                    el *= RAD2DEG

                    state_data.extend([az,el])

                #photometric data
                if 'FLUX_W_M2' and 'FLUX_APPARENT_MAG' in self._valid_features:
                    flux_received, apparent_magnitude, is_visible = self.station.calculate_light_flux(state=final_state,agent=expert)
                    state_data.extend([flux_received,apparent_magnitude])


                all_expert_states[j].append(state_data)


        #extract data
        # y_train = self.scaler.fit_transform(observations[1:,:])
        y_train = observations.astype(float)[1:,np.newaxis,:]
        
        # all_expert_states = np.asarray(all_expert_states)

        x_train_list = []

        # for expert in all_expert_states:
            # expert_np = np.array(expert)
            # x_train_list.append(self.scaler.fit_transform(expert_np))
        x_train_list = np.asarray([expert for expert in all_expert_states])
        
        x_train = np.stack(x_train_list, axis=1)  # shape: (T, N, F)

        residuals = y_train - x_train

        self.scaler.fit(residuals[:,0,:])

        residuals_norm = np.array([self.scaler.transform(res) for res in residuals])

        return x_train, y_train, residuals



        

            









