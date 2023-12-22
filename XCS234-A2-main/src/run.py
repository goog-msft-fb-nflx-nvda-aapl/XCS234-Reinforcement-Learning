import os
import shutil
import sys
import argparse
import gym
import subprocess
from pathlib import Path
import utils
from utils.general import get_logger, join
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from submission.q3_schedule import LinearExploration, LinearSchedule
from submission.q5_linear_torch import Linear
from submission.q6_dqn_torch import NatureQN

import yaml
import warnings

# supress gym warnings
warnings.filterwarnings("ignore", module=r"gym")
yaml.add_constructor("!join", join)

parser = argparse.ArgumentParser(
    description="A program to run assignment 2 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--config_filename",
    help="The name of the config file in the config/ directory to be used for model training.",
    default="q7_dqn",
)
"""
This script enables you to run deep Q network or linear approximation according to a custom config file.
Feel free to change the configurations (in the config/ folder).
If you choose to do so, please report your hyperparmeters. You'll find the results,
log and recording of your agent in the results folder. A good way to monitor the progress
of you model training is to use Tensorboard. Below we have provided some details on how
to setup tensorboard, further instructions can be found in the assignment handout.

To launch tensorboard, open a Terminal window and run
tensorboard --logdir=results/ --host 0.0.0.0
Then, port forward 6006 on the vm (tensorboard's default port)
to an available port on your local machine (e.g. 12345) using the following command
ssh -L 6006:localhost:12345 -p xxxxx student@ml-lab-xxxxxxx.eastus.cloudapp.azure.com
where you replace the "x's" above with the values you use to regularly ssh onto your
Azure lab vm.
"""
if __name__ == "__main__":
    # read in script arguments
    args = parser.parse_args()

    # read config file
    config_file = open("config/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # check environment name in config file
    if config["env"]["env_name"] == "ALE/Pong-v5":
        # make env
        env = gym.make(
            config["env"]["env_name"],
            frameskip=(2, 5),
            full_action_space=False,
            render_mode=config["env"]["render_mode"],
        )
        env = MaxAndSkipEnv(env, skip=config["hyper_params"]["skip_frame"])
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=config["env"]["overwrite_render"],
        )

        # exploration strategy
        exp_schedule = LinearExploration(
            env,
            config["hyper_params"]["eps_begin"],
            config["hyper_params"]["eps_end"],
            config["hyper_params"]["eps_nsteps"],
        )

        # learning rate schedule
        lr_schedule = LinearSchedule(
            config["hyper_params"]["lr_begin"],
            config["hyper_params"]["lr_end"],
            config["hyper_params"]["lr_nsteps"],
        )

        if config["model"] == "dqn":
            # check weights
            load_path = Path(config["model_training"]["load_path"])

            if not load_path.is_file():
                raise RuntimeError(
                    "Pretrained weights not found. Please checkout the repo"
                )

            # train model
            model = NatureQN(env, config)
            model.run(exp_schedule, lr_schedule)

        elif config["model"] == "linear":
            # train model
            model = Linear(env, config)
            model.run(exp_schedule, lr_schedule)

        else:
            sys.exit(
                "Incorrectly specified model, config['model'] should either be dqn or linear."
            )

    elif config["env"]["env_name"] == "test_environment":

        if config["model"] == "dqn":
            env = utils.EnvTest((80, 80, 3))

            # exploration strategy
            exp_schedule = LinearExploration(
                env,
                config["hyper_params"]["eps_begin"],
                config["hyper_params"]["eps_end"],
                config["hyper_params"]["eps_nsteps"],
            )

            # learning rate schedule
            lr_schedule = LinearSchedule(
                config["hyper_params"]["lr_begin"],
                config["hyper_params"]["lr_end"],
                config["hyper_params"]["lr_nsteps"],
            )

            # train model
            model = NatureQN(env, config)
            model.run(exp_schedule, lr_schedule)

        elif config["model"] == "linear":
            env = utils.EnvTest((5, 5, 1))

            # exploration strategy
            exp_schedule = LinearExploration(
                env,
                config["hyper_params"]["eps_begin"],
                config["hyper_params"]["eps_end"],
                config["hyper_params"]["eps_nsteps"],
            )

            # learning rate schedule
            lr_schedule = LinearSchedule(
                config["hyper_params"]["lr_begin"],
                config["hyper_params"]["lr_end"],
                config["hyper_params"]["lr_nsteps"],
            )

            # train model
            model = Linear(env, config)
            model.run(exp_schedule, lr_schedule)

        else:
            sys.exit(
                "Incorrectly specified model, config['model'] should either be 'dqn' or 'linear'."
            )

    else:
        sys.exit(
            "Incorrectly specified environment,  config['model'] should either be 'Pong-v5' or 'linear'."
        )
