import torch.nn as nn

class Loss(nn.Module):
    def __init__(self,):
        super(Loss, self).__init__()
        
    def forward(self, y_pred, y, x_pred, x,sigma_x, sigma_y):
        loss = (((x-x_pred)**2/sigma_x**2).sum(1) + 
                ((y-y_pred)**2/sigma_y**2).sum(1)).mean()
        return loss
    
class L1Loss(nn.Module):
    def __init__(self,):
        super(L1Loss, self).__init__()
        
    def forward(self, y_pred, y):
        loss = ((y-y_pred)**2).sum(1).mean()
        return loss


class NN(nn.Module):
    def __init__(self,input_dim,output_dim=3, hidden=32, layers = 3, dropout=1e-10, sigma = 1.):
        super(NN, self).__init__()
        self.sigma = sigma
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Linear(input_dim, hidden))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.BatchNorm1d(hidden))
        
        for k in range(layers):
            self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.BatchNorm1d(hidden))
            
        self.layers.append(nn.Linear(hidden, output_dim))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.layers):
            x = l(x)
        return x