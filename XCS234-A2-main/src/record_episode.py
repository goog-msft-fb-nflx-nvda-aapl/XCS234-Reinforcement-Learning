# -*- coding:utf-8 -*-

"""
This script runs one episode and record video with the trained model.

    Example: run either with default parameters or with custom parameters
    >>> python ./record_episode.py
    >>> python ./record_episode.py --config_filename q7_dqn
    >>> python ./record_episode.py --output_filename dqn_eisodes
"""

import sys
import gym
import torch
import torch.nn as nn
import yaml
import warnings
import argparse
from pathlib import Path

from utils.general import get_logger, join
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from submission.q6_dqn_torch import NatureQN

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

parser.add_argument(
    "--output_filename",
    help="Recorded file name",
)

parser.add_argument(
    "--model_filename",
    help="The name of the trained model file.",
    default="./submission/model.weights",
)


def record_episode(config: dict, model_file: str, output_file: str = None) -> None:
    """Record episode based on the trained model."""

    # create the environment
    env = gym.make(
        config["env"]["env_name"],
        frameskip=(2, 5),
        full_action_space=False,
        render_mode=config["env"]["render_mode"]
    )

    env = MaxAndSkipEnv(env, skip=config["hyper_params"]["skip_frame"])
    env = PreproWrapper(
        env,
        prepro=greyscale,
        shape=(80, 80, 1),
        overwrite_render=config["env"]["overwrite_render"]
    )

    if output_file is None:
        output_file = "{}/{}".format(config["output"]["record_path"], args.config_filename)
    else:
        output_file = output_file

    output_file_p = Path(output_file)
    video_folder = output_file_p.parents[0]
    name_prefix = output_file_p.name.split(".")[0]

    print("")
    print("=" * 50)
    print("- Recording dir:\t {}".format(video_folder))
    print("- File prefix:\t{}".format(name_prefix))
    print("- Model file:\t{}".format(model_file))
    print("=" * 50)
    print("")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=lambda x: x % 100 == 0,
        name_prefix=name_prefix
    )

    if config["model"] == "dqn":
        # initialize model and load model weights
        model = NatureQN(env, config)
        model.q_network.load_state_dict(torch.load(model_file, map_location="cpu"))
    else:
        sys.exit("Incorrectly specified model, config['model'] should be dqn.")

    # record an episode
    model.evaluate(env, 1)


if __name__ == "__main__":
    # read in script arguments
    args = parser.parse_args()

    # read config file
    config_file = open("config/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    output_file = args.output_filename

    # check environment name in config file
    if config["env"]["env_name"] == "ALE/Pong-v5":
        record_episode(config, args.model_filename, output_file=output_file)
    else:
        sys.exit(
            "Incorrectly specified environment,  config['model'] should either be 'Pong-v5'."
        )
