"""
Evaluate the performance of the trained agents in the MetaDrive environment.
This file runs on the validation env we defined in env.py.

Usage:

    python eval.py --agent-name <agent_name> --log-dir <log_dir> --num-processes <num_processes> --num-episodes-per-processes <num_episodes_per_processes> --seed <seed> --render

Example to evaluate an agent (--agent-name should be the folder name in 'agents/'):

    python eval.py --agent-name="random_agent"

Quick debug:
    python eval.py --agent-name="random_agent" --num-processes=1 --num-episodes-per-processes=1
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
import tqdm

from agents import load_policies
from env import get_validation_env
from utils import make_envs, Timer, step_envs, pretty_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent-name",
    required=True,
    type=str,
    help="The name of the agent to be evaluated, aka the subfolder name in 'agents/'."
)
parser.add_argument(
    "--log-dir",
    default="data/",
    type=str,
    help="The directory where you want to store the data. "
         "Default: ./data/"
)
parser.add_argument(
    "--num-processes",
    default=10,
    type=int,
    help="The number of parallel RL environments. Default: 10"
)
parser.add_argument(
    "--num-episodes-per-processes",
    default=10,
    type=int,
    help="The number of episode to evaluate per process. Default: 10"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The random seed. This has nothing to do with MetaDrive environment. Default: 0"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Whether to launch both the top-down renderer and the 3D renderer. Default: False."
)
args = parser.parse_args()

if __name__ == '__main__':
    # Verify algorithm and config

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    num_processes = args.num_processes
    render = args.render
    if render:
        assert num_processes == 1


    # Create the environment
    def single_env_factory():
        return get_validation_env()


    envs = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_processes,
        asynchronous=True,
    )
    total_episodes_to_eval = args.num_episodes_per_processes * num_processes

    # Instantiate all policies here.
    all_policies = load_policies()
    policy_class_map = {}
    for key, Policy in all_policies.items():
        policy_class_map[key] = Policy

    # We will use the specified agent.
    agent_name = args.agent_name
    if agent_name not in policy_class_map:
        raise ValueError(f"Agent {agent_name} not found in the agents folder {policy_map.keys()}!")
    policy_class = policy_class_map[agent_name]
    policy = policy_class()

    print("==================================================")
    print(f"EVALUATING AGENT {agent_name} (CREATOR: {policy.CREATOR_NAME}, UID: {policy.CREATOR_UID})")
    print("==================================================")

    # Setup some stats helpers
    episode_rewards = np.zeros([num_processes, 1], dtype=float)
    episode_costs = np.zeros([num_processes, 1], dtype=float)
    total_episodes = total_steps = iteration = 0
    last_total_steps = 0

    result_recorder = defaultdict(list)

    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []

    print("Start evaluation!")
    obs = envs.reset()

    if hasattr(policy, "reset"):
        policy.reset()

    with tqdm.tqdm(total=int(total_episodes_to_eval)) as pbar:
        while True:

            cpu_actions = policy(obs)

            # Step the environment
            obs, reward, done, info, masks, total_episodes, \
                total_steps, episode_rewards, episode_costs = step_envs(
                cpu_actions=cpu_actions,
                envs=envs,
                episode_rewards=episode_rewards,
                episode_costs=episode_costs,
                result_recorder=result_recorder,
                total_steps=total_steps,
                total_episodes=total_episodes,
                device="cpu"
            )

            if hasattr(policy, "reset"):
                policy.reset(done_batch=done)

            if render:
                envs.render(mode="topdown")

            pbar.update(total_episodes - pbar.n)
            if total_episodes >= total_episodes_to_eval:
                break

    print("==================================================")
    print(f"THE PERFORMANCE OF {agent_name}:")
    pretty_print({k: np.mean(v) for k, v in result_recorder.items()})
    print("==================================================")

    envs.close()