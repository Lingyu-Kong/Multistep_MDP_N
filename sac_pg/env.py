from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
calc=LennardJones(rc=500)

class Env(object):
    def __init__(
        self,
        if_trunc:bool,
        max_steps:int,
        fmax:float,
    ):
        self.if_trunc=if_trunc
        self.max_steps=max_steps
        self.fmax=fmax

    def compute(self,xyz):
        N=len(xyz)
        atm=Atoms('Ar'+str(N),positions=xyz)
        atm.calc=calc
        if self.if_trunc:
            return min(atm.get_potential_energy(),10)
        else:
            return atm.get_potential_energy()
        
    def relax(self,xyz,relax_steps=None):
        if relax_steps is None:
            relax_steps=self.max_steps
        N=len(xyz)
        atm=Atoms('Ar'+str(N),positions=xyz)
        atm.calc=calc
        dyn = BFGS(atm,logfile=None)
        dyn.run(fmax=self.fmax,steps=relax_steps)
        steps=dyn.get_number_of_steps()
        if self.if_trunc:
            energy=min(dyn.atoms.get_potential_energy(),10)
        else:
            energy=dyn.atoms.get_potential_energy()
        pos=dyn.atoms.get_positions()
        return steps,energy,pos
    
    def if_done(self,xyz):
        N=len(xyz)
        atm=Atoms('Ar'+str(N),positions=xyz)
        atm.calc=calc
        dyn = BFGS(atm,logfile=None)
        dyn.run(steps=1,fmax=self.fmax)
        return dyn.converged()

    def step(self,state,action):
        next_state=state+action
        energy_0=self.compute(state.tolist())
        energy_1=self.compute(next_state.tolist())
        reward=energy_0-energy_1 ## 能量下降值
        done=self.if_done(next_state.tolist())
        # done=False
        return next_state,reward,done
