from sympy import re
import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

shared_params={
    "device":device,
    "num_atoms":10,
    "pos_scale": 2.0,
    "action_scale": 0.1,
}

training_params={
    "num_episodes":100,
    "max_episode_steps":100,
    "start_steps":0,
    "update_per_step":2,
    "batch_size":64,
    "buffer_init_size":2000,
    "imi_num_epochs":500,
}

gnn_params = {
    "device":device,
    "lr":5e-5,
    "num_atoms":shared_params["num_atoms"],
    "mlp_hidden_size":512,
    "mlp_layers":2,
    "latent_size":256,
    "use_layer_norm":False,
    "num_message_passing_steps":6,
    "global_reducer":"sum",
    "node_reducer":"sum",
    "dropedge_rate":0.1,
    "dropnode_rate":0.1,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
    "cycle":1,
    "node_attn":True,
    "global_attn":True,
}

actor_params = {
    "gnn_params":gnn_params,
    "mlp_params":{
        "input_size":gnn_params["latent_size"],
        "output_sizes":[gnn_params["latent_size"]]*2+[3],
        "use_layer_norm":False,
        "activation":nn.ReLU,
        "dropout":0.1,
        "layernorm_before":False,
        "use_bn":False,
    },
    "device":shared_params["device"],
    "lr":5e-5,
    "action_scale":shared_params["action_scale"],
    "decay_steps":200,
    "decay_rate":0.99,
}

critic_params = {
    "gnn_params":gnn_params,
    "mlp_params":{
        "input_size":gnn_params["latent_size"],
        "output_sizes":[gnn_params["latent_size"]]*2+[1],
        "use_layer_norm":False,
        "activation":nn.ReLU,
        "dropout":0.1,
        "layernorm_before":False,
        "use_bn":False,
    },
    "device":shared_params["device"],
    "lr":5e-5,
    "decay_steps":200,
    "decay_rate":0.99,
}

replay_buffer_params = {
    "buffer_size":10000,
    "num_atoms":shared_params["num_atoms"],
}

env_params = {
    "if_trunc":True,
    "max_steps":200,
    "fmax":0.005,
}

sampler_params = {
    "num_atoms":shared_params["num_atoms"],
    "pos_scale":shared_params["pos_scale"],
    "action_scale":shared_params["action_scale"],
    "threshold":1.0,
    "env_params":env_params,
    "max_relax_steps":50,
}

agent_params = {
    "actor_params":actor_params,
    "critic_params":critic_params,
    "sampler_params":sampler_params,
    "replay_buffer_params":replay_buffer_params,
    "env_params":env_params,
    "gamma":0.9,
    "alpha":0.2,
    "entropy_tuning":False,
    "tau":0.01,
    "lr":5e-5,
    "device":device,
    "soft_update_freq":5,
}

wandb_config = {
    "num_atoms":shared_params["num_atoms"],
    "pos_scale": shared_params["pos_scale"],
    "action_scale": shared_params["action_scale"],
    "num_episodes":training_params["num_episodes"],
    "max_episode_steps":training_params["max_episode_steps"],
    "start_steps":training_params["start_steps"],
    "update_per_step":training_params["update_per_step"],
    "batch_size":training_params["batch_size"],
}