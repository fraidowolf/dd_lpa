
from ensemble.Bregressor_ import BaggingRegressor, pre_training
from ensemble import Bregressor_
from ensemble.utils import set_module
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_handling import load_data,Container,Normalize,Denormalize
import os
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io
import pickle


class Loss(nn.Module):
    def __init__(self,):
        super(Loss, self).__init__()
        
    def forward(self, y_pred, y, x_pred, x,sigma_x, sigma_y):
        loss = (((x-x_pred)**2/sigma_x**2).sum(1) + 
                ((y-y_pred)**2/sigma_y**2).sum(1)).mean()
        return loss

class NN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=3, 
                 hidden=32, 
                 layers = 3, 
                 dropout=1e-10, 
                 sigma = 1.,
                 activation=nn.Tanh()):
        
        super(NN, self).__init__()
        self.sigma = sigma
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Linear(input_dim, hidden))
        self.layers.append(activation)
        self.layers.append(nn.BatchNorm1d(hidden))
        
        for k in range(layers):
            self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.BatchNorm1d(hidden))
            
        self.layers.append(nn.Linear(hidden, output_dim))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.layers):
            x = l(x)
        return x
    
def train(config):

    if 'pre_training' in config: Bregressor_.pre_training= config['pre_training']
    if 'lr1' in config: set_module.lr1 = config['lr1']
    if 'lr2' in config: set_module.lr2 = config['lr2']
    if not 'norm' in config: config['norm'] = None
    if not 'cuda' in config: config['cuda'] = False
    
    if not 'activation' in config:
        activation = nn.Tanh()
    elif config['activation'] == 'ReLU':
        activation = nn.ReLU()
    elif config['activation'] == 'Tanh':
        activation = nn.Tanh()

    trainset, testset, norm  = load_data(config['path'],
                                config['inputs'],
                                config['outputs'],
                                samples=config['samples'],
                                ratio=config['ratio'],
                                start_ind = config['start_ind'],
                                norm=config['norm'],
                                random=False)

    trainset = Container(trainset)
    testset = Container(testset)
    
    train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=int(config["batch_size"]),
    shuffle=True,
    num_workers=0)

    test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=int(config["batch_size"]),
    shuffle=True,
    num_workers=0)
    
    dest_path = config['dest_path']
    if dest_path:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        if not os.path.exists('./logs/'+dest_path):
            os.makedirs('./logs/'+dest_path)


    # Define the ensemble
    ensemble = BaggingRegressor(
        estimator=nn.ModuleList(config['model']),                     
        n_estimators=config['estimators'], 
        cuda=config['cuda'],
        n_jobs=1,
    )
    
    ensemble.set_criterion(config['loss_fun'])
    
    # Set the optimizer
    #ensemble.set_optimizer()
        
    if dest_path:
        logger = set_logger(f'{dest_path}/ensamble_net.log')
    

    # Train the ensemble
    ensemble.fit(
        train_loader,
        test_loader = test_loader,
        epochs=config['epochs'],            
        save_model = False,
    )
    
    
    if dest_path:
        io.save(ensemble.to('cpu'), f'./{dest_path}/', logger)
        config['norm'] = norm
    
        with open(f'./{dest_path}/config.pkl', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ensemble
        
        
if __name__ == '__main__':
#2.4620936548071057 2.227688417910039

    path = '/beegfs/desy/group/mls/experiments/lux/user/frida/24h_run/dataframe_combined_espec_interpolated_gaia_energy_2022.h5'
    inputs = ['gaia_energy_interpolated',
              'oap_nf_mean_x',
              'oap_nf_mean_y',
              'oap_nf_peak_signal',
              'amp2_spec_mean',
              'amp2_spec_rms',
              '1',
              '2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8']  
    

    outputs = ['espec_high_energy_median','espec_high_energy_mad','bpm1_q','xspec_1st_order_wavelangth','xspec_1st_order_width']
    
    din = len(inputs)
    dout = len(outputs)
    config = {'path' : path, # Path to data
              'epochs': 300, # training epochs
              'estimators': 100, # networks in ensemble
              'inputs': inputs, # input labels
              'outputs': outputs, # output labels
              'samples': 20000, # number of samples form dataset
              'start_ind': 0, # start ind of samples from dataset 
              'batch_size': 1024, # batch size network
              'ratio': 0.9,
              'loss_fun':  Loss(), # loss function
              'model' : [NN(din+dout,din,hidden=32,layers=2),
                         NN(din,dout,hidden=32,layers=2)], # model
              'dest_path':'./20231204/MJL_with_xspec/', # destination path of model
             }
    train(config)

