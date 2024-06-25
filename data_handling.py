import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn
from sklearn.decomposition import PCA as PCA_
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def PCA_transform(x, pca = None, n_components= 20): 
    if not pca:
        pca = Pipeline([('pca', PCA_(n_components=n_components,random_state=0)), ('scaling', StandardScaler())])
        pca.fit_transform(x)  
        
    return pca.transform(x), pca


def PCA_invtransform(x, pca):
    return pca.inverse_transform(x)


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
              return_index=False,
              pca = None,
              pca_components = 20):

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

    if pca:
        pca_in = None
        pca_out = None 
        for i,pca_col in enumerate(pca):
            if pca_col in inputs:
                x_ = np.array(df_input[pca_col].to_list())
                x_, pca_in = PCA_transform(x_,n_components=pca_components[i])
                for j in range(pca_components[i]):
                    df_input[pca_col+'_'+str(j)] = x_[:,j]
                df_input = df_input.drop(columns=[pca_col])
            if pca_col in outputs:
                x_ = np.array(df_output[pca_col].to_list())
                x_, pca_out = PCA_transform(x_,n_components=pca_components[i])
                for j in range(pca_components[i]):
                    df_output[pca_col+'_'+str(j)] = x_[:,j]
                df_output = df_output.drop(columns=[pca_col])
        columns = [df_input.columns.to_list(),df_output.columns.to_list()]

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
    print(df_input.shape,df_output.shape)

    if norm:
        Xtrain,Xnorm = Normalize(df_input[ind_train],norm[0])
        Ytrain,Ynorm = Normalize(df_output[ind_train],norm[1])
    else:
        Xtrain,Xnorm = Normalize(df_input[ind_train])
        Ytrain,Ynorm = Normalize(df_output[ind_train])
    
    Xtest,_ = Normalize(df_input[ind_val],norm=Xnorm)
    Ytest,_ = Normalize(df_output[ind_val],norm=Ynorm)

    out = [[Xtrain,Ytrain], [Xtest,Ytest], [Xnorm,Ynorm]]
    if return_index:
        out.append([index[ind_train],index[ind_val]])

    if pca:
        out.append([pca_in,pca_out])
        out.append(columns)

    return out   
    
