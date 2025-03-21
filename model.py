import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.nn import Parameter
from types import SimpleNamespace

from ipdb import set_trace
from utilis import glorot_init
from function_laplacian_diffusion import *


class AGCN(nn.Module):
    def __init__(self, num_nodes):
        super(AGCN, self).__init__()
        self.n = num_nodes
        self.w1 = Parameter(torch.FloatTensor(self.n,self.n))
        self.w1.data = torch.eye(self.n)
        self.w2 = Parameter(torch.FloatTensor(self.n,self.n))
        self.w2.data = torch.eye(self.n)
    
    def forward(self, X, A):
        # print(A.size(), X.size())
        H = torch.mm(torch.mm(A, self.w1), A.T) 
        # print(H.size(), X.size()
        H = torch.mm(torch.mm(H, self.w2), X)
        embed = torch.mm(H, H.T)  
        embed = F.normalize(embed, dim=1) 

        return embed


class GCN(nn.Module):
    def __init__(self, input_dim, activation = F.relu, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, input_dim)
        self.activation = activation

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class ConstantODEblock(nn.Module):
    def __init__(self, args, edge_index, edge_weight):
        super(ConstantODEblock, self).__init__()
        self.args = args
        self.t = torch.tensor([0, args.time]).to(args.device)

        self.odefunc = LaplacianODEFunc(args, edge_index, args.device)
        
        self.odefunc.edge_index = edge_index.to(args.device)
        self.odefunc.edge_weight = edge_weight.to(args.device)

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.atol = args.tol_scale * 1e-7 
        self.rtol = args.tol_scale * 1e-9 
        
    def set_x0(self, x0): 
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x) 
        integrator = self.train_integrator if self.training else self.test_integrator
        
        func = self.odefunc
        state = x  
        
        state_dt = integrator( 
            func, state, t,
            method= self.args.method, 
            options = dict(step_size=1, max_iters=100),
            atol=self.atol,   
            rtol=self.rtol)   
        z = state_dt[1]
        return z

class ConstantODEblock1(nn.Module):
    def __init__(self, args):
        super(ConstantODEblock1, self).__init__()
        self.args = args
        self.t = torch.tensor([0, args.time]).to(args.device)

        self.odefunc = LaplacianODEFunc(args, None, args.device)

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.atol = args.tol_scale * 1e-7   
        self.rtol = args.tol_scale * 1e-9  
        
    def set_x0(self, x0):  
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x) 
        integrator = self.train_integrator if self.training else self.test_integrator
        
        func = self.odefunc
        state = x  
        
        state_dt = integrator( 
            func, state, t,
            method= self.args.method, 
            options = dict(step_size=1, max_iters=100),
            atol=self.atol,   
            rtol=self.rtol)   

        z = state_dt[1]
        return z

class SCDGN(nn.Module):
    def __init__(self, N, edge_index, edge_weight,  args):
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        self.n_layers = args.n_layers

        self.AttenGCN = AGCN(N)
        
        self.extractor = nn.ModuleList()
        self.extractor.append(nn.Linear(N, args.hid_dim))
        for i in range(self.n_layers - 1):
            self.extractor.append(nn.Linear(args.hid_dim, args.hid_dim))
        self.dropout = nn.Dropout(p=args.dropout)
        
        self.diffusion = ConstantODEblock(args, edge_index,edge_weight )

        self.init_weights()
        
        self.params_imp = list(self.diffusion.parameters()) 
        self.params_exp = list(self.AttenGCN.parameters())   \
                        + list(self.extractor.parameters()) 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def forward(self, knn, adj, norm_factor):

        h = self.AttenGCN(knn,adj)
        
        for i, layer in enumerate(self.extractor):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)

        self.diffusion.set_x0(h) 
        new_z = self.diffusion(h)   
        z = F.relu(norm_factor * new_z + h)
        z = (z - z.mean(0)) / z.std()
        
        return z
    
class SCDGN1(nn.Module):
    def __init__(self, N, args):
        super().__init__()

        self.n_layers = args.n_layers

        self.AttenGCN = AGCN(N)
        
        self.extractor = nn.ModuleList()
        self.extractor.append(nn.Linear(N, args.hid_dim))
        for i in range(self.n_layers - 1):
            self.extractor.append(nn.Linear(args.hid_dim, args.hid_dim))
        self.dropout = nn.Dropout(p=args.dropout)
        
       

        self.diffusion = ConstantODEblock1(args)

        self.init_weights()
        
        self.params_imp = list(self.diffusion.parameters()) 
        self.params_exp = list(self.AttenGCN.parameters())   \
                        + list(self.extractor.parameters()) 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def forward(self, knn, adj, norm_factor, edge_index, edge_weight, args):
        
        self.diffusion.odefunc.edge_index = edge_index.to(args.device)
        self.diffusion.odefunc.edge_weight = edge_weight.to(args.device)

        h = self.AttenGCN(knn,adj)

        
        for i, layer in enumerate(self.extractor):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)

        self.diffusion.set_x0(h)
        new_z = self.diffusion(h)   

        z = F.relu(norm_factor * new_z + h)
        z = (z - z.mean(0)) / z.std()

        
        return z
