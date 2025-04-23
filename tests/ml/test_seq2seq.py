from python_propagate.ml.models.lstm import BayesianLSTM, LSTM
from python_propagate.ml.preprocess.load import load_data_from_h5
from python_propagate.ml.preprocess.preprocess import (NonDimensional,Seq2SeqDataModule)

from python_propagate.ml.models.seq2seq import Seq2Seq, duration_to_steps

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler, Normalizer,RobustScaler
import numpy as np
import matplotlib.pyplot as plt

"""
NOTE: 

it would be interesting to look at at the start of the filter phase,
predict based on the prior, and the preceeding predictions depend on the
previous estimates. I could also then batch like this (1,0->1) (2, ((0->1,0->2))
"""

def plot_residuals(y_test, y_predicted, save_path='tests/ml/results/geo_test1/residuals.png'):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    for i, ax in enumerate(axs):
        residual = (y_test[:, :, i] - y_predicted[:, :, i]).flatten() * 50000  # Denormalize
        ax.plot(residual, color='purple')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel(f"Residual (dim {i})")
        ax.grid(True)

    axs[-1].set_xlabel("Timestep")
    fig.suptitle("Residuals (Truth - Prediction)")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_trajectory(y_test,y_predicted):

    fig, axs = plt.subplots(3,1,figsize = (10,8))

    for i, ax in enumerate(axs):
        ax.plot(y_predicted[:,:,i].flatten()*50000,color = 'red', marker = '*', label = "Predicted")
        ax.plot(y_test[:,:,i].flatten()*50000,color = 'blue',label = 'Truth')

    plt.savefig('tests/ml/results/geo_test1/trajectory.png')



def plot_test(train_loader):

    # Get a single example
    x_test, y_test = next(iter(train_loader))
    x_test = x_test[:1]
    y_test = y_test[:1]

    pred = model.predict_variable_horizon(x_test, duration_seconds=10 * 300)  # 10 steps at 30s interval
    pred = pred.squeeze().cpu().detach().numpy()
    truth = y_test.squeeze().numpy()

    import matplotlib.pyplot as plt
    plt.plot(pred, label='Predicted')
    plt.plot(truth, label='Truth')
    plt.legend()
    plt.title("Sine Prediction")
    plt.savefig('tests/ml/results/geo_test1/test.png')
    # plt.show()


def test_seq2seq_lstm(train_loader,scaler,n_epcochs = 10):

    # Example usage
    # model = BayesianLSTM(input_size=6, hidden_size=32, output_size=6)
    model = Seq2Seq(input_size=6, hidden_size=64, output_size=6)
    # model = Seq2Seq(input_size=1, hidden_size=64, output_size=1)
    model = torch.compile(model)

    model.train_model(train_loader=train_loader,n_epochs=n_epcochs,scaler=scaler)

    return model

def test_predict(model,x_test,y_test):


    # prediction = model.predict(x_test)
    predictions = model.predict_variable_horizon(x_test,duration_seconds = y_test.shape[1]*30)

    return predictions.cpu().detach().numpy()



def test_plot(predictions, y_test):

    plot_trajectory(y_test=y_test,y_predicted=predictions)
    plot_residuals(y_test=y_test,y_predicted=predictions)


    return None
def load_data(agent_name = 'GEO1'):

    # data = load_data_from_h5(file = 'tests/ml/results/geo_test1/MLForge.h5')
    data = load_data_from_h5(file = 'tests/ml/results/geo_test1/MLForge1hrdt.h5')

    return data[agent_name]

def preprocess(data):
    scaler = NonDimensional(sma=50000)
    # scaler = MinMaxScaler(feature_range=(0,1))

    train_days = 20
    total_seconds = data.shape[0] * 30  # assuming 30s intervals
    total_days = total_seconds / 86400

    # train_index = int(train_days * 86400 / 30)
    #dt is 1 hour
    train_index = int(train_days * 86400 / 3600)

    # Fit scaler on training set only
    # data = scaler.fit_transform(data)
    data[:, 1:4] = scaler.fit_position(data[:, 1:4])
    data[:, 4:7] = scaler.fit_velocity(data[:, 4:7])

    # === TRAIN SET ===
    train_data = data[:train_index, 1:7]
    train_module = Seq2SeqDataModule(
        data=train_data,
        input_window=48,
        target_window=48,
        overlap=0.5,
        batch_size=64,
        shuffle=True,
    )
    train_loader = train_module.get_dataloader()

    # === TEST SET ===
    test_data = data[train_index:, 1:7]
    test_module = Seq2SeqDataModule(
        data=test_data,
        input_window=24,
        target_window=24,
        overlap=0.0,  # no overlap for simplicity
        batch_size=1,
        shuffle=False
    )
    test_loader = test_module.get_dataloader()

    
    return train_loader, test_loader, scaler



if __name__ == "__main__":

    # test_create_lstm()
    data = load_data()

    train_loader, test_loader, scalar = preprocess(data)
    # train_loader = DataLoader(SineWaveDataset(), batch_size=64, shuffle=True)

    x_test, y_test = next(iter(test_loader))
    # x_test, y_test = next(iter(train_loader))
    # x_test = x_test.detach().numpy()
    # y_test = y_test.detach().numpy()
    model = test_seq2seq_lstm(train_loader,scalar,2000)
    predictions = test_predict(model, x_test[0].unsqueeze(0), y_test[0].unsqueeze(0))
    # plot_test(train_loader=train_loader)
    # idx = duration_to_steps(duration_seconds=86400)
    # predictions = predictions

    # test_plot(predictions=predictions,y_test=y_test[:,:idx,:])
    test_plot(predictions=predictions,y_test=y_test)


