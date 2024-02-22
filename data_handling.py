import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn


class DISCNN(nn.Module):
    def __init__(self,input_dim,output_dim=1,hidden=32,layers = 2):
        super(DISCNN, self).__init__()
        self.flatten = nn.Flatten()
        
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden))
        self.layers.append(nn.ReLU())
        
        for k in range(layers):
            self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden, output_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.layers):
            x = l(x)
        return x

    



def Normalize(x,norm=None):
    if not norm:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    else:
        mean = norm[0]
        std = norm[1]
    return (x - mean)/std,[mean,std] 


def Denormalize(x,norm):
    mean = norm[0]
    std = norm[1]
    return x*std + mean


class Container(Dataset):
    def __init__(self, data):   
        
        self.x = data[0]
        self.y = data[1]
        
        if torch.cuda.is_available():
            self.x = torch.tensor(self.x).float().to('cuda:0')
            self.y = torch.tensor(self.y).float().to('cuda:0')
        else:
            self.x = torch.tensor(self.x).float()
            self.y = torch.tensor(self.y).float() 
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i): 
        return self.x[i], self.y[i]
    
    
def random_choice(N,ratio = 0.8, seed=0):
    inds = np.arange(N)
    nratio = int(N*ratio)
    
    rng = np.random.default_rng(seed)
    rng.shuffle(inds)
    train = inds[:nratio]
    val = inds[nratio:]
    return train,val
    
    
def load_data(path, inputs, outputs, 
              ratio=0.5, 
              samples=10000, 
              start_ind=0,
              val_shotnr=None, 
              norm=None, 
              random=False,
              return_index=False):

    df = pd.read_hdf(path)[start_ind:start_ind+samples]
    
    if 'bpm2_q' in df.columns:
        q_f = (df['bpm2_q'] > 20) 
        ff = df['amp2_energy'] > 5.5
        fff = df['saga3_energy'] > 1
        ffff = df['oap_ff_peak_x']> -400
        
        df_input = df[inputs]
        df_output = df[outputs]
        
        nonan1 = ~df_input.isna().any(axis=1)
        nonan2 = ~df_output.isna().any(axis=1)
        f = nonan1 & nonan2 & q_f & ff & fff & ffff
        
    else:
        df_input = df[inputs]
        df_output = df[outputs]  
        nonan1 = ~df_input.isna().any(axis=1)
        nonan2 = ~df_output.isna().any(axis=1)
        f = nonan1 & nonan2 
        
    index = df.index[f].values
    df_input = df_input[f]
    df_output = df_output[f]
        
    if random:
        ind_train,ind_val = random_choice(df_input.shape[0],ratio)
    elif val_shotnr:
        ind = df_input.loc[:val_shotnr].shape[0]
        ind_train,ind_val = np.arange(0,ind),np.arange(ind,df_input.shape[0])
    else:
        ind = int(df_input.shape[0]*ratio)
        ind_train,ind_val = np.arange(0,ind),np.arange(ind,df_input.shape[0])
    
    df_input = df_input.values
    df_output = df_output.values
    
    if norm:
        Xtrain,Xnorm = Normalize(df_input[ind_train],norm[0])
        Ytrain,Ynorm = Normalize(df_output[ind_train],norm[1])
    else:
        Xtrain,Xnorm = Normalize(df_input[ind_train])
        Ytrain,Ynorm = Normalize(df_output[ind_train])
    
    Xtest,_ = Normalize(df_input[ind_val],norm=Xnorm)
    Ytest,_ = Normalize(df_output[ind_val],norm=Ynorm)

    if return_index:
        return [Xtrain,Ytrain], [Xtest,Ytest], [Xnorm,Ynorm], [index[ind_train],index[ind_val]]
    else:
        return [Xtrain,Ytrain], [Xtest,Ytest], [Xnorm,Ynorm]
    
    
