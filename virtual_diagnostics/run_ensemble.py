import sys
sys.path.append('../')
from train_ensemble import train, NN, Loss
from data_handling import Container,load_data,Normalize,Denormalize


        
        
if __name__ == '__main__':

    path = '../data/dataframe_combined_xspec_espec_interpolated_gaia_energy_2022.h5'

    inputs = [#'espec_high_energy_spectrum_interp',
              'bpm2_q',
              'bpm2_x',
              'bpm2_y',
              'gaia_energy_interpolated',
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
              '8',] 
    

    outputs = ['xspec_spectrum',]
    
    din = len(inputs)
    dout = len(outputs)
    pca_components = 20
    config = {'path' : path, # Path to data
              'epochs': 500, # training epochs
              'pre_training' : 500, # pre training epochs out of epochs, i.e all epochs are with MSE loss
              'estimators': 2, # networks in ensemble
              'inputs': inputs, # input labels
              'outputs': outputs, # output labels
              'samples': 40000, # number of samples form dataset
              'start_ind': 0, # start ind of samples from dataset 
              'batch_size': 1024, # batch size network
              'ratio': 0.5,
              'loss_fun':  Loss(), # loss function
              'model' : [NN(din+dout+pca_components-1,din,hidden=64,layers=2), #not used
                         NN(din,dout+pca_components-1,hidden=64,layers=2)], # model
              'activation' : 'ReLU',
              'dest_path':'../models/VD_BPM/', # destination path of model
              'lr1' : 0.0001,
              'lr2' : 0.0001,
              'cuda' : True,
              'pca' : ['xspec_spectrum'],
              'pca_components' : [20],
             }
    train(config)
'''
[NN(din+dout+pca_components*2-2,din+pca_components-1,hidden=64,layers=2), #not used
                         NN(din+pca_components-1,dout+pca_components-1,hidden=64,layers=2)], # model
'''