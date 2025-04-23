from python_propagate.ml.models.lstm import BayesianLSTM, LSTM
from python_propagate.ml.preprocess.load import load_data_from_h5
from python_propagate.ml.preprocess.preprocess import (NonDimensional,
                                                       create_overlapping_batches,
                                                        create_trajectory_dataset,
                                                        create_trajectory_batch,
                                                        OverlappingDataset,
                                                        BatchDataset,Seq2SeqDataModule)

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

"""
NOTE: 

it would be interesting to look at at the start of the filter phase,
predict based on the prior, and the preceeding predictions depend on the
previous estimates. I could also then batch like this (1,0->1) (2, ((0->1,0->2))
"""

def plot_residuals(y_test,y_predicted):

    fig, axs = plt.subplots(3,1,figsize = (10,8))

    for i, ax in enumerate(axs):
        res = y_test[:,i] - y_predicted[:,i]
        ax.plot(res)

    plt.savefig('tests/ml/results/geo_test1/residuals.png')

def plot_trajectory(y_test,y_predicted):

    fig, axs = plt.subplots(3,1,figsize = (10,8))

    for i, ax in enumerate(axs):
        ax.plot(y_predicted[:,i],color = 'red', marker = '*', label = "Predicted")
        ax.plot(y_test[:,i],color = 'blue',label = 'Truth')

    plt.savefig('tests/ml/results/geo_test1/trajectory.png')

    


def test_train_lstm(train_loader, x_test, y_test):

    # Example usage
    # model = BayesianLSTM(input_size=6, hidden_size=32, output_size=6)
    model = LSTM(input_size=6, hidden_size=32, output_size=6)

    model.train_model(train_loader=train_loader,n_epochs=100)

    return model

def test_predict(model,x_test):


    # prediction = model.predict(x_test)
    predictions = model.predict_sequential(x_test[:1],timesteps = 8000)

    return predictions



def test_plot(predictions, y_test):

    # model.eval()
    # predictions = []
    # device = next(model.parameters()).device
    # h = torch.zeros(1, model.hidden_size, device=device)
    # c = torch.zeros(1, model.hidden_size, device=device)
    # for i in range(len(x_test)):
    #     x_t = torch.as_tensor(x_test[np.newaxis, i:i+1, :], dtype=torch.float32, device=device)
    #     h, c = model.cell(x_t.squeeze(1), (h, c))
    #     y_predict = model.fc(h)
    #     predictions.append(y_predict.cpu().detach().numpy())
    #     print(f'Predictions: {(i / len(x_test)) * 100:.2f}%')
    # predictions = np.squeeze(np.array(predictions))
    # plot_residuals(y_test, predictions)

    # res = y_test[0] - y_predict.detach().numpy()[0,0]
    
    # weight_mu = model.fc.weight_mu.detach().cpu().numpy()
    # weight_logvar = model.fc.weight_logvar.detach().cpu().numpy()

    # mu = weight_mu.mean(axis=1)
    # std = (np.exp(0.5*weight_logvar)).mean(axis=1)
    # predictions.to('cpu')
    # plot_residuals(np.squeeze(y_test*50000), predictions*50000)
    # plot_trajectory(np.squeeze(y_test*50000), predictions*50000)
    predictions = np.squeeze(predictions)
    plot_trajectory(y_test[0:8000]*50000, predictions*50000)



    return None
def load_data(agent_name = 'GEO1'):

    data = load_data_from_h5(file = 'tests/ml/results/geo_test1/MLForge.h5')

    return data[agent_name]

def preprocess(data):

    scaler = NonDimensional(sma = 50000)

    train_days = 7
    day_index = int(train_days * 86400 / 30)

    # data[:,0] = scaler.fit_time(data[:,0])
    data[:,1:4] = scaler.fit_position(data[:,1:4])
    data[:,4:7] = scaler.fit_velcoity(data[:,4:7])



    # x_train, y_train = create_trajectory_dataset(data[:day_index,1:4],data[:day_index,4:7],data[0,0])
    x_train, y_train = create_trajectory_batch(data[:day_index,1:7],window_size=100,overlap=0.6,dt = 30)
    # x_test, y_test = create_trajectory_dataset(data[day_index:,1:4],data[day_index:,4:7],data[0,0])
    # x_test, y_test = create_trajectory_batch(data[day_index:,1:7],window_size=2000,overlap=0.6,dt = 30)
    # _, truth = create_trajectory_batch(data[day_index:,1:7],window_size=2000,overlap=0.6,dt = 30)
    # truth = data[day_index+1:,1:7]
    x_test = data[day_index:,1:7]
    y_test = data[day_index + 1:,1:7]
    


    
    module = Seq2SeqDataModule(data=data[:day_index,1:7],input_window=1000,target_window=1000,overlap=0.5,batch_size=32)
    data_batches = BatchDataset(inputs=x_train,targets=y_train)

    # train_loader = DataLoader(data_batches, batch_size=1, shuffle=True)
    train_loader = module.get_dataloader()
    # train_loader = DataLoader(data_batches, batch_size=1, shuffle=True)


    return train_loader, x_test, y_test


if __name__ == "__main__":

    # test_create_lstm()
    data = load_data()

    train_loader, x_test, y_test = preprocess(data)


    model = test_train_lstm(train_loader=train_loader,x_test=x_test, y_test=y_test)

    predictions = test_predict(model, torch.as_tensor(x_test[np.newaxis,:,:],dtype=torch.float64))

    # predictions = predictions

    test_plot(predictions=predictions,y_test=y_test)


