import torch
import torch.nn.functional as F
from torch.optim import Adam
import time
import wandb
import matplotlib.pyplot as plt

from sac_pg.actor import Actor
from sac_pg.critic import Critic
from sac_pg.sampler import Sampler
from sac_pg.replay_buffer import ReplayBuffer
from sac_pg.env import Env

from utils.tensor_utils import to_tensor

"""
Policy Gradient Agent based on SAC
"""

class Agent(object):
    def __init__(
        self,
        actor_params:dict,
        critic_params:dict,
        sampler_params:dict,
        replay_buffer_params:dict,
        env_params:dict,
        gamma:float,
        alpha:float,
        entropy_tuning:bool,
        tau:float,
        lr:float,
        device:torch.device,
        soft_update_freq:int,
    ):
        self.actor=Actor(**actor_params)
        self.critic=Critic(**critic_params)
        self.critic_target=Critic(**critic_params)
        self.sampler=Sampler(**sampler_params)
        self.replay_buffer=ReplayBuffer(**replay_buffer_params)
        self.env=Env(**env_params)
        self.gamma=gamma
        self.alpha=alpha
        self.entropy_tuning=entropy_tuning
        self.tau=tau
        self.soft_update_freq=soft_update_freq
        if self.entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([10,3]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def train(
        self,
        num_episodes:int,
        max_episode_steps:int,
        start_steps:int,
        update_per_step:int,
        batch_size:int,
        buffer_init_size:int,
        imi_num_epochs:int,
    ):
        self.enhanced_buffer_init(buffer_init_size)
        self.imitation_learning(imi_num_epochs,batch_size)
        update_time_count=0
        for episode_i in range(num_episodes):
            episode_steps=0
            episode_reward=0
            done=False
            state=self.sampler.single_conform_sample()
            start_time=time.time()
            episode_path=[]
            while not done:
                if episode_steps<start_steps:
                    action=self.sampler.single_action_sample()
                else:
                    action,_,_=self.actor.sample(to_tensor(state).unsqueeze(0))
                    action=action.squeeze(0).detach().cpu().numpy()
                
                if self.replay_buffer.buffer_top>=batch_size:
                    for i in range(update_per_step):
                        critic1_loss,critic2_loss,actor_loss,entropy_loss,alpha=self.update_parameters(batch_size,update_time_count)
                        update_time_count+=1
                        wandb.log({
                            "critic1_loss":critic1_loss,
                            "critic2_loss":critic2_loss,
                            "actor_loss":actor_loss,
                            "entropy_loss":entropy_loss,
                            "alpha":alpha,
                        })
                    
                next_state,reward,done=self.env.step(state,action)
                episode_reward+=reward
                episode_path.append(episode_reward)
                episode_steps+=1
                mask=0 if episode_steps==max_episode_steps else float(not done)
                done=done or (episode_steps==max_episode_steps)
                self.replay_buffer.store(state,action,reward,next_state,mask)
                state=next_state
            end_time=time.time()
            plt.figure()
            plt.plot(episode_path,label="episode_path")
            plt.legend()
            wandb.log({
                "episode_path "+str(episode_i):plt,
            })    
            print("Episode: {}, Reward: {}, Steps: {}, Time: {}".format(episode_i,episode_reward,episode_steps,end_time-start_time))
    
    def update_parameters(
        self,
        batch_size,
        update_time_count
    ):
        state_batch,action_batch,reward_batch,next_state_batch,mask_batch=self.replay_buffer.sample(batch_size)
        state_batch=to_tensor(state_batch)
        action_batch=to_tensor(action_batch)
        reward_batch=to_tensor(reward_batch)
        next_state_batch=to_tensor(next_state_batch)
        mask_batch=to_tensor(mask_batch)
        ## critic update
        with torch.no_grad():
            next_action_batch,next_log_prob,_=self.actor.sample(next_state_batch)
            q1_next,q2_next=self.critic_target(next_state_batch,next_action_batch)
            q_target=(torch.min(q1_next,q2_next)-self.alpha*next_log_prob)*self.gamma*mask_batch+reward_batch
        q1,q2=self.critic(state_batch,action_batch)
        q1_loss=F.mse_loss(q1,q_target)
        q2_loss=F.mse_loss(q2,q_target)
        q_loss=q1_loss+q2_loss
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        self.critic.scheduler.step()
        ## actor update
        pi,log_prob,_=self.actor.sample(state_batch)
        q1_pi,q2_pi=self.critic(state_batch,pi)
        q_pi=torch.min(q1_pi,q2_pi)
        actor_loss=((self.alpha*log_prob)-q_pi).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.scheduler.step()
        ## alpha update
        if self.entropy_tuning:
            alpha_loss=-(self.log_alpha*(log_prob+self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha=self.log_alpha.exp()
        ## soft update
        if update_time_count%self.soft_update_freq==0:
            self.soft_update(self.critic_target,self.critic,self.tau)
        if self.entropy_tuning:
            return q1_loss.item(),q2_loss.item(),actor_loss.item(),alpha_loss.item(),self.alpha.item()
        else:
            return q1_loss.item(),q2_loss.item(),actor_loss.item(),0,0

    def imitation_learning(self,num_epochs,batch_size):
        for i in range(num_epochs):
            start_time=time.time()
            state_batch,action_batch,reward_batch,next_state_batch,mask_batch=self.replay_buffer.sample(batch_size)
            state_batch=to_tensor(state_batch)
            action_batch=to_tensor(action_batch)
            reward_batch=to_tensor(reward_batch)
            next_state_batch=to_tensor(next_state_batch)
            mask_batch=to_tensor(mask_batch)
            ## actor imitation learning
            _,_,pi=self.actor.sample(state_batch)
            action_batch=action_batch.view(batch_size,-1)
            pi=pi.view(batch_size,-1)
            actor_loss=F.mse_loss(action_batch,pi)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.actor.scheduler.step()
            end_time=time.time()
            wandb.log({"imit_actor_loss":actor_loss.item()})
            print("Epoch: {}, Actor Loss: {}, Time: {}".format(i,actor_loss.item(),end_time-start_time))

    def enhanced_buffer_init(self,buffer_init_size):
        for _ in range(buffer_init_size):
            state=self.sampler.single_conform_sample()
            action,reward,next_state,done=self.sampler.enhanced_sample(state)
            self.replay_buffer.store(state,action,reward,next_state,done)
    
    def soft_update(self,target,source,tau):
        for target_param,source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_(target_param.data*(1.0-tau)+source_param.data*tau)
    
    def hard_update(self,target,source):
        for target_param,source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_(source_param.data)

    