from platform import node
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as Dist
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from gnn.ignn import IGNN
from gnn.metalayer import MLPwoLastAct

epsilon=1e-6

"""
Gaussian Policy Network
"""

class Actor(nn.Module):
    def __init__(
        self,
        gnn_params:dict,
        mlp_params:dict,
        device:float,
        lr:float,
        action_scale:float,
        decay_steps:int,
        decay_rate:float,
    ):
        super().__init__()
        self.action_scale=action_scale
        self.gnn=IGNN(**gnn_params)
        self.mu_net=MLPwoLastAct(**mlp_params)
        self.sigma_net=MLPwoLastAct(**mlp_params)
        self.device=device
        self.mu_net.to(self.device)
        self.sigma_net.to(self.device)
        self.gnn.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.scheduler=StepLR(self.optimizer,decay_steps,decay_rate)
        self.weight_init(self.mu_net)
        self.weight_init(self.sigma_net)
        self.weight_init(self.gnn)

    def forward(self,state):
        node_attr,_,_=self.gnn(state)
        mu=self.mu_net(node_attr)
        sigma=self.sigma_net(node_attr)
        return mu,sigma
    
    def sample(self,state):
        mu,log_sigma=self.forward(state)
        sigma=log_sigma.exp()
        normal=Dist.Normal(mu,sigma)
        x=normal.rsample()
        y=torch.tanh(x)
        action=y*self.action_scale
        log_prob=normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + epsilon)
        log_prob = log_prob.view(state.shape[0],-1).sum(1, keepdim=True)
        mean=torch.tanh(mu) * self.action_scale
        action=action.view(state.shape[0],state.shape[1],-1)
        return action,log_prob,mean ##[batch_size,num_atoms,3],[batch_size,1],[batch_size,num_atoms,3]
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def save_model(self,path):
        torch.save(self.state_dict(),path)

    def load_model(self,path):
        self.load_state_dict(torch.load(path))