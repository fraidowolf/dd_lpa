import sys
sys.path.append('../')
from train_ensemble import train, NN, Loss
from data_handling import Container,load_data,Normalize,Denormalize


        
        
if __name__ == '__main__':

    path = '../data/dataframe_combined_espec_interpolated_gaia_energy_2022.h5'
    all_inputs = ['gaia_energy_interpolated',
              'oap_nf_peak_x',
              'oap_nf_fwhm_x',
              'oap_nf_mean_x',
              'oap_nf_rms_x',
              'oap_nf_curve_max_x',
              'oap_nf_signal_in_fwhm_x',
              'oap_nf_signal_in_rms_x',
              'oap_nf_peak_y',
              'oap_nf_fwhm_y',
              'oap_nf_mean_y',
              'oap_nf_rms_y',
              'oap_nf_curve_max_y',
              'oap_nf_signal_in_fwhm_y',
              'oap_nf_signal_in_rms_y',
              'oap_nf_peak_signal',
              'oap_ff_peak_x',
              'oap_ff_fwhm_x',
              'oap_ff_mean_x',
              'oap_ff_rms_x',
              'oap_ff_curve_max_x',
              'oap_ff_signal_in_fwhm_x',
              'oap_ff_signal_in_rms_x',
              'oap_ff_peak_y',
              'oap_ff_fwhm_y',
              'oap_ff_mean_y',
              'oap_ff_rms_y',
              'oap_ff_curve_max_y',
              'oap_ff_signal_in_fwhm_y',
              'oap_ff_signal_in_rms_y',
              'oap_ff_peak_signal',
              'amp2_spec_peak',
              'amp2_spec_max',
              'amp2_spec_fwhm',
              'amp2_spec_fwhm_signal',
              'amp2_spec_mean',
              'amp2_spec_rms',
              'amp2_spec_rms_signal',
              '1',
              '2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8',
              '9',
              '10',
              '11',
              '12',
              '13',
              '14',
              '15',
              '16',
              '17',
              '18',
              '19',
              '20',
              '21',
              '22',
              '23',
              '24',
              '25',
              '26',
              '27',
              '28',
              '29',
              '30',
              '31',
              '32']    
    

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
    

    outputs = ['espec_high_energy_median','espec_high_energy_mad','bpm1_q']
    
    din = len(inputs)
    dout = len(outputs)
    config = {'path' : path, # Path to data
              'epochs': 200, # training epochs
              'pre_training' : 200, # pre training epochs out of epochs, i.e all epochs are with MSE loss
              'estimators': 2, # networks in ensemble
              'inputs': inputs, # input labels
              'outputs': outputs, # output labels
              'samples': 40000, # number of samples form dataset
              'start_ind': 0, # start ind of samples from dataset 
              'batch_size': 1024, # batch size network
              'ratio': 0.5,
              'loss_fun':  Loss(), # loss function
              'model' : [NN(din+dout,din,hidden=64,layers=2), #not used
                         NN(din,dout,hidden=64,layers=2)], # model
              'activation' : 'ReLU',
              'dest_path':'./models/RELU/', # destination path of model
              'lr1' : 0.0001,
              'lr2' : 0.0001,
              'cuda' : True,
             }
    train(config)

