from abc import ABC
from typing import Union, List, Optional, Type
from copy import deepcopy

import torch
from torch import nn
import numpy as np

from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
import gymnasium as gym

from .net import ActorCritic

class BasePolicy(ABC, nn.Module):
    """The base class for safe RL policy.

    The base class follows a similar structure as `Tianshou
    <https://github.com/thu-ml/tianshou>`_. All of the policy classes must inherit
    :class:`~fsrl.policy.BasePolicy`.

    A policy class typically has the following parts:

    * :meth:`~srl.policy.BasePolicy.__init__`: initialize the policy, including coping
        the target network and so on;
    * :meth:`~srl.policy.BasePolicy.forward`: compute action with given observation;
    * :meth:`~srl.policy.BasePolicy.process_fn`: pre-process data from the replay buffer
        (this function can interact with replay buffer);
    * :meth:`~srl.policy.BasePolicy.learn`: update policy with a given batch of data.
    * :meth:`~srl.policy.BasePolicy.post_process_fn`: update the replay buffer from the
        learning process (e.g., prioritized replay buffer needs to update the weight);
    * :meth:`~srl.policy.BasePolicy.update`: the main interface for training, i.e.,
        `process_fn -> learn -> post_process_fn`.

    Most of the policy needs a neural network to predict the action and an optimizer to
    optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a ``torch.Tensor``, a \
    dict or any others), hidden state "state" (for RNN usage), and other information \
    "info" provided by the environment. 2. Output: some "logits", the next hidden state
    "state", and the intermediate result during policy forwarding procedure "policy". The
    "logits" could be a tuple instead of a ``torch.Tensor``. It depends on how the policy
    process the network output. For example, in PPO, the return of the network might be
    ``(mu, sigma), state`` for Gaussian policy. The "policy" can be a Batch of
    torch.Tensor or other things, which will be stored in the replay buffer, and can be
    accessed in the policy update process (e.g. in "policy.learn()", the "batch.policy"
    is what you need).

    Since :class:`~fsrl.policy.BasePolicy` inherits ``torch.nn.Module``, you can use
    :class:`~fsrl.policy.BasePolicy` almost the same as ``torch.nn.Module``, for
    instance, loading and saving the model: ::

        torch.save(policy.state_dict(), "policy.pth")
        policy.load_state_dict(torch.load("policy.pth"))

    :param torch.nn.Module actor: the actor network.
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param dist_fn: distribution class for stochastic policy to sample the action.
        Default to None :type dist_fn: Type[torch.distributions.Distribution]
    :param BaseLogger logger: the logger instance for logging training information. \
        Default to DummyLogger.
    :param float gamma: the discounting factor for cost and reward, should be in [0, 1].
        Default to 0.99.
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: normalize estimated values to have std close to 1,
        also normalize the advantage to Normal(0, 1). Default to False.
    :param deterministic_eval: whether to use deterministic action instead of stochastic
        action sampled by the policy. Default to True.
    :param action_scaling: whether to map actions from range [-1, 1] to range \
        [action_spaces.low, action_spaces.high]. Default to True.
    :param action_bound_method: method to bound action to range [-1, 1]. Default to
        "clip".
    :param observation_space: environment's observation space. Default to None.
    :param action_space: environment's action space. Default to None.
    :param lr_scheduler: learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Optional[Type[torch.distributions.Distribution]] = None,
        gamma: float = 0.99,
        max_batchsize: Optional[int] = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        if isinstance(critics, nn.Module):
            self.critics = nn.ModuleList([critics])
        elif isinstance(critics, List):
            self.critics = nn.ModuleList(critics)
        else:
            raise TypeError("critics should not be %s" % (type(critics)))
        self.critics_old = deepcopy(self.critics)
        self.critics_num = len(self.critics)
        self.dist_fn = dist_fn
        assert 0.0 <= gamma <= 1.0, "discount factor should be in [0, 1]."
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        self._max_batchsize = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critics)

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_type = ""
        if isinstance(action_space, (Discrete, MultiDiscrete, MultiBinary)):
            self.action_type = "discrete"
        elif isinstance(action_space, Box):
            self.action_type = "continuous"
        else:
            print("Warning! The action sapce type is unclear, regard it as continuous.")
            self.action_type = "continuous"
            print(self.action_space)
        self.updating = False
        self.action_scaling = action_scaling
        # can be one of ("clip", "tanh", ""), empty string means no bounding
        assert action_bound_method in ("", "clip", "tanh")
        self.action_bound_method = action_bound_method
        self.gradient_steps = 0


    def map_action(self, act):
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer and
        thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it to
        [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act