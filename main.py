import sac_pg.config as config
from sac_pg.agent import Agent
import wandb

agent_params=config.agent_params
training_params=config.training_params
wandb_config=config.wandb_config

wandb.login()
wandb.init(project="sac_pg",config=wandb_config)

if __name__=="__main__":
    agent=Agent(**agent_params)
    agent.train(**training_params)
    agent.actor.save_model("./sac_pg/model_save/actor_final.pt")
    agent.critic.save_model("./sac_pg/model_save/critic_final.pt")