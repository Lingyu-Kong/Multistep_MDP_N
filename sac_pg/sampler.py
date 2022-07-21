import numpy as np
from sac_pg.env import Env

class Sampler(object):
    def __init__(
        self,
        num_atoms:int,
        pos_scale:float,
        action_scale:float,
        threshold:float,
        env_params:dict,
        max_relax_steps:int,
    ):
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.action_scale=action_scale
        self.threshold=threshold
        self.env=Env(**env_params)
        self.max_relax_steps=max_relax_steps

    def single_conform_sample(self):
        pos=np.zeros((self.num_atoms,3))
        for i in range(self.num_atoms):
            if_continue=True
            while if_continue:
                new_pos=np.random.rand(3)*2*self.pos_scale-self.pos_scale
                if_continue=False
                for j in range(i):
                    distance=np.linalg.norm(new_pos-pos[j],ord=2)
                    if distance<self.threshold:
                        if_continue=True
                        break
            pos[i,:]=new_pos
        return pos

    def single_action_sample(self):
        return np.random.rand(self.num_atoms,3)*2*self.action_scale-self.action_scale

    def enhanced_sample(self,state):
        relax_steps=self.max_relax_steps*np.random.rand()
        _,_,next_state=self.env.relax(state.tolist(),relax_steps)
        action=np.array(next_state)-state
        energy_0=self.env.compute(state.tolist())
        energy_1=self.env.compute(next_state.tolist())
        reward=energy_0-energy_1
        done=self.env.if_done(next_state.tolist())
        return action,reward,next_state,done
