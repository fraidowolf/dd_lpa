import numpy as np
import pandas as pd


def generate(f,x):
    x_esp = np.random.normal(0,0.5,x.shape)
    y_esp = np.random.normal(0,0.5,x.shape)
    return x+x_esp,f(x)+y_esp
  
    
    
if __name__=='__main__':
    n = 3000
    x = np.random.normal(0,1,n)
    
    f = lambda x : 0.5*x**2 
    x_exp,y_exp = generate(f,x)
    
    data = {}
    data['x'] = x_exp
    data['y'] = y_exp
    path = './toydata.h5'
    df = pd.DataFrame(data)
    df.to_hdf(path, key='df', mode='w')