from offline_RL.ensemble_dqn import EnsembleAgent
from offline_RL.rem_dqn import REMAgent
import wandb
import argparse
from omegaconf import OmegaConf 


def run_exp():

    wandb_run = wandb.init(
        project="EnsemblevsREM_DQN"
    )
    wandb.define_metric("train/loss", step_metric="train/step")
    wandb.define_metric("eval/reward", step_metric="eval/step")
    wandb.define_metric("eval/epl", step_metric="eval/step")


    config = wandb.config
    agent = EnsembleAgent(config, wandb_run)
    agent.reset()
    agent.train_agent()
    agent.evaluate_agent()  

    agent = REMAgent(config, wandb_run)
    agent.reset()
    agent.train_agent()
    agent.evaluate_agent() 



if __name__ == "__main__":
    """
    Parse the config file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True )

    """
    Create the wandb run
    """

    project_name = "EnsemblevsREM_DQN"
    sweep_id = wandb.sweep(sweep=config_dict, project=project_name)
    agent = wandb.agent(sweep_id, function=run_exp, count=5)
