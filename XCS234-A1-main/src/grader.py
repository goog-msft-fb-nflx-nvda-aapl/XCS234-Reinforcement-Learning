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

# Import student submission
import submission


#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

DET4 = [
    0.59,
    0.656,
    0.729,
    0.656,
    0.656,
    0.0,
    0.81,
    0.0,
    0.729,
    0.81,
    0.9,
    0.0,
    0.0,
    0.9,
    1.0,
    0.0,
]  # True values of states in Deterministic-4x4-FrozenLake-v1
HOLES4 = [5, 7, 11, 12, 15]  # Location of holes in Deterministic-4x4-FrozenLake-v1

### BEGIN_HIDE ###
### END_HIDE ###


def _policy_performance(env, algorithm, num_episodes, gamma=0.9, tol=1e-3):
    """
    Runs RL algorithm for a number of episodes and returns the average reward per episode.
    """
    _, final_p = algorithm(
        env.P, env.observation_space.n, env.action_space.n, gamma, tol
    )
    policy = final_p.astype(int)
    sum_rewards = 0
    for _ in range(num_episodes):
        episode_reward = 0
        state, info = env.reset()
        terminated, truncated = False, False
        count = 0
        while not (terminated or truncated) and count < 10000:
            action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            count += 1
        sum_rewards += episode_reward
    return float(sum_rewards) / float(num_episodes)


def mask_policy4(policy):
    """
    Masks the location of holes in the Deterministic-4x4-FrozenLake-v0 environment.
    """
    for i in HOLES4:
        policy[i] = 0


### BEGIN_HIDE ###
### END_HIDE ###


#########
# TESTS #
#########


class Test_4a(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """4a-0-basic: Test policy evaluation on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        policy = np.array([1, 2, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 3, 2, 2, 0])
        true_value_function = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.81,
            0.0,
            0.0,
            0.0,
            0.9,
            0.0,
            0.0,
            0.9,
            1.0,
            0.0,
        ]
        value_function = submission.policy_evaluation(
            env.P,
            env.observation_space.n,
            env.action_space.n,
            policy,
            gamma=0.9,
            tol=1e-3,
        )
        for i in range(len(value_function)):
            self.assertAlmostEqual(
                value_function[i], true_value_function[i], delta=0.001
            )

    ### BEGIN_HIDE ###
### END_HIDE ###


class Test_4b(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """4b-0-basic: Test policy improvement on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        val = np.array(
            [
                0.063,
                0.0,
                0.071,
                0.0,
                0.086,
                0.0,
                0.11,
                0.0,
                0.141,
                0.244,
                -0.297,
                0.0,
                0.0,
                0.378,
                0.638,
                0.0,
            ]
        )
        policy = np.array([0, 3, 0, 3, 1, 0, 2, 0, 2, 1, 0, 0, 1, 2, 1, 1])
        true_policy = [1, 2, 1, 0, 1, 0, 3, 0, 2, 1, 1, 0, 0, 2, 2, 0]
        mask_policy4(true_policy)
        new_policy = submission.policy_improvement(
            env.P, env.observation_space.n, env.action_space.n, val, policy, gamma=0.9
        ).tolist()
        mask_policy4(new_policy)
        self.assertListEqual(new_policy, true_policy)

    ### BEGIN_HIDE ###
### END_HIDE ###


class Test_4c(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """4c-0-basic: Test PI values on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        final_v, _ = submission.policy_iteration(
            env.P, env.observation_space.n, env.action_space.n, 0.9, 1e-3
        )
        for i in range(len(final_v)):
            self.assertAlmostEqual(final_v[i], DET4[i], delta=0.001)

    @graded(timeout=1, is_hidden=False)
    def test_1(self):
        """4c-1-basic: Test PI performance on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        performance = _policy_performance(env, submission.policy_iteration, 1)
        self.assertGreaterEqual(performance, 0.999)

    ### BEGIN_HIDE ###
### END_HIDE ###


class Test_4d(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """4d-0-basic: Test values from VI on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        final_v, _ = submission.value_iteration(
            env.P, env.observation_space.n, env.action_space.n, 0.9, 1e-3
        )
        for i in range(len(final_v)):
            self.assertAlmostEqual(final_v[i], DET4[i], delta=0.001)

    @graded(timeout=1, is_hidden=False)
    def test_1(self):
        """4d-1-basic: Test performance of VI on 4x4 deterministic environment"""
        env = gym.make("Deterministic-4x4-FrozenLake-v1")
        performance = _policy_performance(env, submission.value_iteration, 1)
        self.assertGreaterEqual(performance, 0.999)

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
