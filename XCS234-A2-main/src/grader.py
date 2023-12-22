#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import gym
import numpy as np
import yaml
import warnings

import utils
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
import torch
import torch.nn as nn

# supress gym warnings
warnings.filterwarnings("ignore", module=r"gym")

# Import student submission
import submission

# Import configuration settings
def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)

q5_config_file = open("config/q5_linear.yml")
q5_config = yaml.load(q5_config_file, Loader=yaml.FullLoader)

q6_config_file = open("config/q6_dqn_grader.yml")
q6_config = yaml.load(q6_config_file, Loader=yaml.FullLoader)

q7_config_file = open("config/q7_dqn_grader.yml")
q7_config = yaml.load(q7_config_file, Loader=yaml.FullLoader)

# Ensure tests run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = " "


#########
# TESTS #
#########


class Test_3a(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """3a-0-basic: Check Exploration Strategy 1"""
        random.seed(1000)
        env = utils.EnvTest((5, 5, 1))
        exp_strat = submission.LinearExploration(env, 1, 0, 10)
        found_diff = False
        for i in range(10):
            rnd_act = exp_strat.get_action(0)
            if rnd_act != 0 and rnd_act is not None:
                found_diff = True
        self.assertEqual(found_diff, True)

    @graded(timeout=1, is_hidden=False)
    def test_1(self):
        """3a-1-basic: Check Exploration Strategy 2"""
        env = utils.EnvTest((5, 5, 1))
        exp_strat = submission.LinearExploration(env, 1, 0, 10)
        exp_strat.update(4)
        self.assertEqual(exp_strat.epsilon, 0.6)

    ### BEGIN_HIDE ###
### END_HIDE ###


class Test_5b(GradedTestCase):
    @graded(timeout=8, is_hidden=False)
    def test_0(self):
        """5b-0-basic: Tests for model configurations"""
        env = utils.EnvTest((5, 5, 1))
        model = submission.Linear(env, q5_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n
        self.assertTrue(isinstance(model.q_network, nn.Linear))
        self.assertEqual(
            model.q_network.weight.size(),
            torch.Size(
                [
                    num_actions,
                    img_height
                    * img_width
                    * n_channels
                    * q5_config["hyper_params"]["state_history"],
                ]
            ),
        )
        self.assertTrue(isinstance(model.target_network, nn.Linear))
        self.assertEqual(
            model.target_network.weight.size(),
            torch.Size(
                [
                    num_actions,
                    img_height
                    * img_width
                    * n_channels
                    * q5_config["hyper_params"]["state_history"],
                ]
            ),
        )


### BEGIN_HIDE ###
### END_HIDE ###


class Test_5e(GradedTestCase):
    @graded(timeout=4, is_hidden=False)
    def test_0(self):
        """5e-0-basic: Test for correctly calculating the loss"""
        env = utils.EnvTest((5, 5, 1))
        model = submission.Linear(env, q5_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n

        nn.init.constant_(model.q_network.weight, 0.7)
        nn.init.constant_(model.q_network.bias, 0.7)
        nn.init.constant_(model.target_network.weight, 0.2)
        nn.init.constant_(model.target_network.bias, 0.2)
        state = torch.full(
            (
                2,
                img_height,
                img_width,
                n_channels * q5_config["hyper_params"]["state_history"],
            ),
            0.5,
        )

        with torch.no_grad():
            q_values = model.get_q_values(state, "q_network")
            target_q_values = model.get_q_values(state, "target_network")
        actions = torch.tensor([1, 3], dtype=torch.int)
        rewards = torch.tensor([5, 5], dtype=torch.float)
        terminated_mask = torch.tensor([0, 0], dtype=torch.bool)
        truncated_mask = torch.tensor([0, 0], dtype=torch.bool)

        q_values[0,[0,2,3]] -= 1
        q_values[1,[0,1,2]] -= 1
        loss = model.calc_loss(q_values, target_q_values, actions, rewards, terminated_mask, truncated_mask)
        self.assertEquals(round(loss.item(), 1), 424.4)


class Test_5f(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """5f-0-basic: Test for adding the correct optimizer"""
        env = utils.EnvTest((5, 5, 1))
        model = submission.Linear(env, q5_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n

        model.add_optimizer()
        self.assertTrue(isinstance(model.optimizer, torch.optim.Adam))
        self.assertEqual(len(model.optimizer.param_groups), 1)
        self.assertTrue(
            model.optimizer.param_groups[0]["params"][0] is model.q_network.weight
        )
        self.assertTrue(
            model.optimizer.param_groups[0]["params"][1] is model.q_network.bias
        )


### BEGIN_HIDE ###
### END_HIDE ###


class Test_6a(GradedTestCase):
    @graded(timeout=4, is_hidden=False)
    def test_0(self):
        """6a-0-basic: Tests input and output shapes"""
        env = utils.EnvTest((80, 80, 1))
        model = submission.NatureQN(env, q6_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n
        sample_input = torch.randn(
            1,
            img_height,
            img_width,
            n_channels * q6_config["hyper_params"]["state_history"],
        )

        self.assertTrue(model.q_network, nn.Sequential)
        self.assertTrue(any([isinstance(x, nn.Linear) for x in model.q_network]))
        self.assertTrue(any([isinstance(x, nn.ReLU) for x in model.q_network]))
        self.assertTrue(any([isinstance(x, nn.Flatten) for x in model.q_network]))

    ### BEGIN_HIDE ###
### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
