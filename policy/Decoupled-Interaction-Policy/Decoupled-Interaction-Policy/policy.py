import os
import sys
sys.path.insert(0, './policy/Decoupled-Interaction-Policy/Decoupled-Interaction-Policy')

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDIPWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg):
    workspace = TrainDIPWorkspace(cfg)
    workspace.eval()

class DIP:
    def __init__(self, cfg, checkpoint_num) -> None:
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, checkpoint_num)
        
    def update_obs(self, observation):
        self.env_runner.update_obs(observation)
    
    def get_action(self, observation):
        action = self.env_runner.get_action(self.policy, observation)
        return action    

    def get_policy_and_runner(self, cfg, checkpoint_num):
        workspace = TrainDIPWorkspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, checkpoint_num)
        return policy, env_runner

if __name__ == "__main__":
    main()
