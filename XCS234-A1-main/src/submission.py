import argparse
import numpy as np
import gym
import time
from gym.envs.toy_text import frozen_lake
from gym.envs.registration import register

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    help="The name of the environment to run your algorithm on.",
    choices=[
        "Deterministic-4x4-FrozenLake-v1",
        "Stochastic-4x4-FrozenLake-v1",
        "Deterministic-8x8-FrozenLake-v1",
        "Stochastic-8x8-FrozenLake-v1",
    ],
    default="Deterministic-4x4-FrozenLake-v1",
)

parser.add_argument(
    "--algorithm",
    help="The name of the algorithm to run.",
    choices=["both", "policy_iteration", "value_iteration"],
    default="both",
)

# Register custom gym environments
env_dict = gym.envs.registration.registry.copy()
for env in env_dict:
    if "Deterministic-4x4-FrozenLake-v1" in env:
        del gym.envs.registration.registry[env]

    elif "Deterministic-8x8-FrozenLake-v1" in env:
        del gym.envs.registration.registry[env]

    elif "Stochastic-4x4-FrozenLake-v1" in env:
        del gym.envs.registration.registry[env]

    elif "Stochastic-8x8-FrozenLake-v1" in env:
        del gym.envs.registration.registry[env]

register(
    id="Deterministic-4x4-FrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

register(
    id="Deterministic-8x8-FrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": False},
)

register(
    id="Stochastic-4x4-FrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)

register(
    id="Stochastic-8x8-FrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": True},
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P (dict): From gym.core.Environment
		For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a
		list of tuples of the form [(probability, nextstate, reward, terminal),...] where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS (int): number of states in the environment
	nA (int): number of actions in the environment
	gamma (float): Discount factor. Number in range [0, 1)
"""

############################################################
# Problem 4: Frozen Lake MDP
############################################################

############################################################
# Problem 4a: policy evaluation


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """
    Evaluate the value function from a given policy.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            policy (np.array[nS]): The policy to evaluate. Maps states to actions.
            tol (float): Terminate policy evaluation when max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): 
            The value function of the given policy, 
            where value_function[s] is the value of state s.
    """

    value_function = np.zeros(nS)

    ### START CODE HERE ###
    prev_value_function = np.copy( value_function )
    while True:
        for state, action in enumerate(policy):
            immediate_reward = 0
            sum_of_future_reward = 0
            for probability, nextstate, reward, terminal in P[state][action]:
                immediate_reward += probability * reward
                sum_of_future_reward += probability * prev_value_function[nextstate]
            value_function[state] = immediate_reward + gamma * sum_of_future_reward
        # Terminate policy evaluation when max |value_function(s) - prev_value_function(s)| < tol
        if np.max( np.abs(value_function - prev_value_function)) < tol:
            break
        prev_value_function = np.copy( value_function )
    ### END CODE HERE ###
    return value_function


############################################################
# Problem 4b: policy improvement


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            value_from_policy (np.ndarray): The value calculated from the policy
            policy (np.array): The previous policy

    Returns:
            new_policy (np.ndarray[nS]): An array of integers. Each integer is the optimal
            action to take in that state according to the environment dynamics and the
            given value function.
    """

    new_policy = np.zeros(nS, dtype="int")
                    
    ### START CODE HERE ###
    for state in range(nS):
        state_action_value_list = np.zeros(nA)
        for action in range(nA):
            immediate_reward = 0
            sum_of_future_reward = 0
            for probability, nextstate, reward, terminal in P[state][action]:
                immediate_reward += probability * reward
                sum_of_future_reward += probability * value_from_policy[nextstate]
            state_action_value_list[action] = immediate_reward + gamma * sum_of_future_reward
        new_policy[state] = np.argmax(state_action_value_list)
    ### END CODE HERE ###
    return new_policy


############################################################
# Problem 4c: policy iteration


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            tol (float): tol parameter used in policy_evaluation()

    Returns:
            value_function (np.ndarray[nS]): value function resulting from policy iteration
            policy (np.ndarray[nS]): policy resulting from policy iteration

    Hint:
            You should call the policy_evaluation() and policy_improvement() methods to
            implement this method.
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ### START CODE HERE ###
    while True:
		# MDP V function policy evaluation of pi_{i}
        value_function = policy_evaluation( P, nS, nA, policy, gamma, tol )
		
        # pi_{i+1} <--- Policy improvement
        policy_improved = policy_improvement( P, nS, nA, value_function, policy, gamma )
        
        if np.sum(policy_improved != policy) == 0:
            break
        
        policy = policy_improved
		
    ### END CODE HERE ###
    return value_function, policy


############################################################
# Problem 4d: value iteration


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment

    Args:
            P, nS, nA, gamma: defined at beginning of file
            tol (float): Terminate value iteration when
                            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): value function resulting from value iteration
            policy (np.ndarray[nS]): policy resulting from value iteration

    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ### START CODE HERE ###
    # page 49 out module 2 slides
    # loop until convergence
    
    while True:
        prev_value_function = np.copy(value_function)
        for state in range(nS):
            # Bellman backup operator
            value_ = np.zeros(nA)
            for action in range(nA):
                immediate_reward = 0
                sum_of_future_reward = 0
                for probability, nextstate, reward, terminal in P[state][action]:
                    immediate_reward += probability * reward
                    sum_of_future_reward += probability * prev_value_function[nextstate]
                value_[action] = immediate_reward + gamma * sum_of_future_reward
            value_function[state] = np.max(value_)
            policy[state] = np.argmax(value_)
        if np.max( np.abs(value_function - prev_value_function)) < tol:
            break
        prev_value_function = np.copy( value_function )
		
    ### END CODE HERE ###
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Args:
            env (gym.core.Environment): Environment to play on. Must have nS, nA, and P as
              attributes.
            Policy (np.array[env.nS]): The action to take at a given state
    """
    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, terminated, truncated, _ = env.step(a)
        episode_reward += rew
        if terminated or truncated:
            break
    env.render()
    if not (terminated or truncated):
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script arguments
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode="human")

    if (args.algorithm == "both") | (args.algorithm == "policy_iteration"):
        print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

        V_pi, p_pi = policy_iteration(
            env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-3
        )
        render_single(env, p_pi, 100)

    if (args.algorithm == "both") | (args.algorithm == "value_iteration"):

        print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

        V_vi, p_vi = value_iteration(
            env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-3
        )
        render_single(env, p_vi, 100)
