import sys
sys.path.append('../')
from train_ensemble import train, NN, Loss
from data_handling import Container,load_data,Normalize,Denormalize


        
        
if __name__ == '__main__':

    path = '../data/dataframe_combined_espec_interpolated_gaia_energy_2022.h5'
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
              '8',] 
    

    outputs = ['espec_high_energy_median','espec_high_energy_mad','bpm1_q', 'xspec_1st_order_wavelangth','xspec_1st_order_width']
    
    din = len(inputs)
    dout = len(outputs)
    config = {'path' : path, # Path to data
              'epochs': 300, # training epochs
              'pre_training' : 100, # pre training epochs out of epochs, i.e all epochs are with MSE loss
              'estimators': 100, # networks in ensemble
              'inputs': inputs, # input labels
              'outputs': outputs, # output labels
              'samples': 40000, # number of samples form dataset
              'start_ind': 0, # start ind of samples from dataset 
              'batch_size': 1024, # batch size network
              'ratio': 0.5,
              'loss_fun':  Loss(), # loss function
              'model' : [NN(din+dout,din,hidden=32,layers=2), #not used
                         NN(din,dout,hidden=32,layers=2)], # model
              'activation' : 'Tanh',
              'dest_path':'../models/', # destination path of model
              'lr1' : 0.001,
              'lr2' : 0.001,
              'cuda' : True,
             }
    train(config)

