import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from gnn.ignn import IGNN
from gnn.metalayer import MLPwoLastAct

"""
Critic with built-in double-head
"""

class Critic(nn.Module):
    def __init__(
        self,
        gnn_params:dict,
        mlp_params:dict,
        device:torch.device,
        lr:float,
        decay_steps:int,
        decay_rate:float,
    ):
        super().__init__()
        self.gnn1=IGNN(**gnn_params)
        self.global_linear1=MLPwoLastAct(**mlp_params)
        self.gnn2=IGNN(**gnn_params)
        self.global_linear2=MLPwoLastAct(**mlp_params)
        
        self.device=device
        self.global_linear1.to(self.device)
        self.global_linear2.to(self.device)
        self.gnn1.to(self.device)
        self.gnn2.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.scheduler=StepLR(self.optimizer,decay_steps,decay_rate)
        self.weight_init(self.global_linear1)
        self.weight_init(self.global_linear2)
        self.weight_init(self.gnn1)
        self.weight_init(self.gnn2)
    
    def forward(self,state,action):
        conform=state+action
        _,_,global_attr1=self.gnn1(conform)
        q_value1=self.global_linear1(global_attr1)
        _,_,global_attr2=self.gnn2(conform)
        q_value2=self.global_linear2(global_attr2)
        return q_value1,q_value2 ## [batch_size,1]

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def save_model(self,path):
        torch.save(self.state_dict(),path)
    
    def load_model(self,path):
        self.load_state_dict(torch.load(path))