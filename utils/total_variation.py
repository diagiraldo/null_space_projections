import torch
import torch.nn as nn

class TotalVariation(nn.Module):
    
    def __init__(
        self, 
    ):
        super(TotalVariation, self).__init__()
        
    def forward(self, x):
        
        x_dim = x.shape[2:]
        grad_x = torch.gradient(x.view(tuple(x_dim)))
        tv = torch.stack([torch.square(g) for g in grad_x]).sum()
            
        return tv