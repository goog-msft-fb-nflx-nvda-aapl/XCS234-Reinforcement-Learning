import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from utils.test_env import EnvTest

############################################################
# Problem 3: Tabular Q-Learning
############################################################

############################################################
# Problem 3a: learning rate scheduler


class LinearSchedule(object):
    """
    Sets linear schedule for exploration parameter epsilon.

    Args:
            eps_begin (float): initial exploration
            eps_end (float): end exploration
            nsteps (int): number of steps between the two values of eps
    """

    def __init__(self, eps_begin, eps_end, nsteps):

        assert (
            eps_begin >= eps_end
        ), "Epsilon begin ({}) needs to be greater than equal to end ({})".format(
            eps_begin, eps_end
        )

        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates epsilon.

        Args:
                t (int): frame number
        """
        ### START CODE HERE ###
        # For t > self.nsteps self.epsilon remains constant as the last updated self.epsilon value, which is self.eps_end.
        if t > self.nsteps:
            self.epsilon = self.eps_end
        else:
            # for t = 0,           self.epsilon = self.eps_begin
            # for t = self.nsteps, self.epsilon = self.eps_end
            self.epsilon = np.linspace( start=self.eps_begin, stop=self.eps_end, num=self.nsteps+1, endpoint=True )[t]
        ### END CODE HERE ###


############################################################
# Problem 3b: e-greedy exploration


class LinearExploration(LinearSchedule):
    """
    Implements e-greedy exploration with linear decay.

    Args:
            env (object): gym environment
            eps_begin (float): initial exploration rate
            eps_end (float): final exploration rate
            nsteps (int): number of steps taken to linearly decay eps_begin to eps_end
    """

    def __init__(self, env, eps_begin, eps_end, nsteps):
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action.

        Args:
                best_action (int): best action according some policy

        Returns:
                (int) action    
        """
        ### START CODE HERE ###
        if np.random.rand() <= self.epsilon:
            # With probability self.epsilon, return a random action
            # you can access the environment via self.env; 
            # you may use env.action_space.sample() to generate a random action
            return self.env.action_space.sample()
        else:
            # else, return best_action
            return best_action
        ### END CODE HERE ###


def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


def your_test():
    """
    Use this to implement your own tests if you'd like (not required)
    """
    pass


if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()
