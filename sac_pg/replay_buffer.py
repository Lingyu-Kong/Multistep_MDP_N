import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        buffer_size:int,
        num_atoms:int,
    ):
        self.buffer_size=buffer_size
        self.buffer_top=0
        self.state_buffer=np.zeros((buffer_size,num_atoms,3))
        self.action_buffer=np.zeros((buffer_size,num_atoms,3))
        self.reward_buffer=np.zeros((buffer_size,1))
        self.next_state_buffer=np.zeros((buffer_size,num_atoms,3))
        self.mask_buffer=np.zeros((buffer_size,1))

    def store(self,state,action,reward,next_state,mask):
        self.state_buffer[self.buffer_top%self.buffer_size,:,:]=state
        self.action_buffer[self.buffer_top%self.buffer_size,:,:]=action
        self.reward_buffer[self.buffer_top%self.buffer_size,:]=reward
        self.next_state_buffer[self.buffer_top%self.buffer_size,:,:]=next_state
        self.mask_buffer[self.buffer_top%self.buffer_size,:]=mask
        self.buffer_top+=1
    
    def sample(self,batch_size):
        choices=np.random.choice(min(self.buffer_size,self.buffer_top),batch_size)
        state_batch=self.state_buffer[choices,:,:]
        action_batch=self.action_buffer[choices,:,:]
        reward_batch=self.reward_buffer[choices,:]
        next_state_batch=self.next_state_buffer[choices,:,:]
        mask_batch=self.mask_buffer[choices,:]
        return state_batch,action_batch,reward_batch,next_state_batch,mask_batch

