from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, Dict, Union, List, Tuple, Optional, Type, no_type_check
from copy import copy, deepcopy
import warnings

import torch
from torch import nn
import numpy as np

ModuleType = type[nn.Module]
ArgsType = tuple[Any, ...] | dict[Any, Any] | Sequence[tuple[Any, ...]] | Sequence[dict[Any, Any]]
TActionShape: TypeAlias = Sequence[int] | int | np.int64
TLinearLayer: TypeAlias = Callable[[int, int], nn.Module]
T = TypeVar("T")


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: ModuleType | None = None,
    norm_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    activation: ModuleType | None = None,
    act_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    linear_layer: TLinearLayer = nn.Linear,
) -> list[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and activation."""
    layers: list[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]
        else:
            layers += [norm_layer(output_size)]
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
    return layers

class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param input_dim: dimension of the input vector.
    :param output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        device: str | int | torch.device | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim, *list(hidden_sizes)]
        model = []
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
            strict=True,
        ):
            model += miniblock(in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)

TRecurrentState = TypeVar("TRecurrentState", bound=Any)

class NetBase(nn.Module, Generic[TRecurrentState], ABC):
    """Interface for NNs used in policies."""

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: TRecurrentState | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, TRecurrentState | None]:
        pass


class Net(NetBase[Any]):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param softmax: whether to apply a softmax layer over the last layer's
        output.
    :param concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module constructor, which takes the input
        and output dimension as input, as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: int | Sequence[int],
        action_shape: TActionShape = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        device: str | int | torch.device = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
        linear_layer: TLinearLayer = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.Q: MLP | None = None
        self.V: MLP | None = None

        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim,
            output_dim,
            hidden_sizes,
            norm_layer,
            norm_args,
            activation,
            act_args,
            device,
            linear_layer,
        )
        if self.use_dueling:  # dueling DQN
            assert dueling_param is not None
            kwargs_update = {
                "input_dim": self.model.output_dim,
                "device": self.device,
            }
            # Important: don't change the original dict (e.g., don't use .update())
            q_kwargs = {**dueling_param[0], **kwargs_update}
            v_kwargs = {**dueling_param[1], **kwargs_update}

            q_kwargs["output_dim"] = 0 if concat else action_dim
            v_kwargs["output_dim"] = 0 if concat else num_atoms
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim
        else:
            self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """
        logits = self.model(obs)
        batch_size = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            assert self.Q is not None
            assert self.V is not None
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(batch_size, -1, self.num_atoms)
                v = v.view(batch_size, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(batch_size, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state



SIGMA_MIN = -20
SIGMA_MAX = 2



class BaseActor(nn.Module, ABC):
    @abstractmethod
    def get_preprocess_net(self) -> nn.Module:
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        # TODO: ALGO-REFACTORING. Marked to be addressed as part of Algorithm abstraction.
        #  Return type needs to be more specific
        pass


def getattr_with_matching_alt_value(obj: Any, attr_name: str, alt_value: T | None) -> T:
    """Gets the given attribute from the given object or takes the alternative value if it is not present.
    If both are present, they are required to match.

    :param obj: the object from which to obtain the attribute value
    :param attr_name: the attribute name
    :param alt_value: the alternative value for the case where the attribute is not present, which cannot be None
        if the attribute is not present
    :return: the value
    """
    v = getattr(obj, attr_name)
    if v is not None:
        if alt_value is not None and v != alt_value:
            raise ValueError(
                f"Attribute '{attr_name}' of {obj} is defined ({v}) but does not match alt. value ({alt_value})",
            )
        return v
    else:
        if alt_value is None:
            raise ValueError(
                f"Attribute '{attr_name}' of {obj} is not defined and no fallback given",
            )
        return alt_value


def get_output_dim(module: nn.Module, alt_value: int | None) -> int:
    """Retrieves value the `output_dim` attribute of the given module or uses the given alternative value if the attribute is not present.
    If both are present, they must match.

    :param module: the module
    :param alt_value: the alternative value
    :return: the value
    """
    return getattr_with_matching_alt_value(module, "output_dim", alt_value)


class Actor(BaseActor):
    """Simple actor network that directly outputs actions for continuous action space.
    Used primarily in DDPG and its variants. For probabilistic policies, see :class:`~ActorProb`.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            self.output_dim,
            hidden_sizes,
            device=self.device,
        )
        self.max_action = max_action

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: s_B -> action_values_BA, hidden_state_BH | None.

        Returns a tensor representing the actions directly, i.e, of shape
        `(n_actions, )`, and a hidden state (which may be None).
        The hidden state is only not None if a recurrent net is used as part of the
        learning algorithm (support for RNNs is currently experimental).
        """
        action_BA, hidden_BH = self.preprocess(obs, state)
        action_BA = self.max_action * torch.tanh(self.last(action_BA))
        return action_BA, hidden_BH


def setstate(
    cls,
    obj,
    state: Dict[str, Any],
    renamed_properties: Dict[str, Union[str, Tuple[str, Callable[[Dict[str, Any]], Any]]]] = None,
    new_optional_properties: List[str] = None,
    new_default_properties: Dict[str, Any] = None,
    removed_properties: List[str] = None,
) -> None:
    """
    Helper function for safe implementations of __setstate__ in classes, which appropriately handles the cases where
    a parent class already implements __setstate__ and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, __setstate__ is not implemented in object, rendering super().__setstate__(state) invalid in the general case.

    :param cls: the class in which you are implementing __setstate__
    :param obj: the instance of cls
    :param state: the state dictionary
    :param renamed_properties: can be used for renaming as well as for assigning new values.
        If passed must map an old property name to either a new property name or
        to tuple of a new property name and a function that computes the new value from the state dictionary.
    :param new_optional_properties: a list of names of new property names, which, if not present, shall be initialised with None
    :param new_default_properties: a dictionary mapping property names to their default values, which shall be added if they are not present
    :param removed_properties: a list of names of properties that are no longer being used
    """
    # handle new/changed properties
    if renamed_properties is not None:
        # `new` can either be a string or a tuple of a string and a function
        for old_name, new in renamed_properties.items():
            if old_name in state:
                if isinstance(new, str):
                    new_name, new_value = new, state[old_name]
                else:
                    new_name, new_value = new[0], new[1](state)

                del state[old_name]
                state[new_name] = new_value

    if new_optional_properties is not None:
        for mNew in new_optional_properties:
            if mNew not in state:
                state[mNew] = None
    if new_default_properties is not None:
        for mNew, mValue in new_default_properties.items():
            if mNew not in state:
                state[mNew] = mValue
    if removed_properties is not None:
        for p in removed_properties:
            if p in state:
                del state[p]
    # call super implementation, if any
    s = super(cls, obj)
    if hasattr(s, '__setstate__'):
        s.__setstate__(state)
    else:
        obj.__dict__ = state


class CriticBase(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""


class Critic(CriticBase):
    """Simple critic network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.
    :param linear_layer: use this module as linear layer.
    :param flatten_input: whether to flatten input data for the last layer.
    :param apply_preprocess_net_to_obs_only: whether to apply `preprocess_net` to the observations only (before
        concatenating with the action) - and without the observations being modified in any way beforehand.
        This allows the actor's preprocessing network to be reused for the critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        flatten_input: bool = True,
        apply_preprocess_net_to_obs_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        self.apply_preprocess_net_to_obs_only = apply_preprocess_net_to_obs_only
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def __setstate__(self, state: dict) -> None:
        setstate(
            Critic,
            self,
            state,
            new_default_properties={"apply_preprocess_net_to_obs_only": False},
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        if self.apply_preprocess_net_to_obs_only:
            obs, _ = self.preprocess(obs)
        obs = obs.flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        if not self.apply_preprocess_net_to_obs_only:
            obs, _ = self.preprocess(obs)
        return self.last(obs)


class ActorProb(BaseActor):
    """Simple actor network that outputs `mu` and `sigma` to be used as input for a `dist_fn` (typically, a Gaussian).

    Used primarily in SAC, PPO and variants thereof. For deterministic policies, see :class:`~Actor`.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action logits.
    :param unbounded: whether to apply tanh activation on final logits.
    :param conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    # TODO: force kwargs, adjust downstream code
    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,
                self.output_dim,
                hidden_sizes,
                device=self.device,
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        if info is None:
            info = {}
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critics: Union[List, nn.Module]):
        super().__init__()
        self.actor = actor
        if isinstance(critics, List):
            critics = nn.ModuleList(critics)
        self.critics = critics


class DoubleCritic(nn.Module):
    """Double critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net1: a self-defined preprocess_net which output a flattened hidden
        state.
    :param preprocess_net2: a self-defined preprocess_net which output a flattened hidden
        state.
    :param hidden_sizes: a sequence of int for constructing the MLP after preprocess_net.
        Default to empty sequence (where the MLP now contains only a single linear
        layer).
    :param int preprocess_net_output_dim: the output dimension of preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer. Default
        to True.

    For advanced usage (how to customize the network), please refer to tianshou's \
        `build_the_network tutorial <https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network>`_.

    .. seealso::

        Please refer to tianshou's `Net <https://tianshou.readthedocs.io/en/master/api/tianshou.utils.html#tianshou.utils.net.common.Net>`_
        class as an instance of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net1: nn.Module,
        preprocess_net2: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess1 = preprocess_net1
        self.preprocess2 = preprocess_net2
        self.output_dim = 1
        input_dim = getattr(preprocess_net1, "output_dim", preprocess_net_output_dim)
        self.last1 = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
        self.last2 = deepcopy(self.last1)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> list:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits1, hidden = self.preprocess1(obs)
        logits1 = self.last1(logits1)
        logits2, hidden = self.preprocess2(obs)
        logits2 = self.last2(logits2)
        return [logits1, logits2]

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, list]:
        """Mapping: (s, a) -> logits -> Q(s, a).

        :return: q value, and a list of two q values (used for Bellman backup)"""
        q_list = self(obs, act, info)
        q = torch.min(q_list[0], q_list[1])
        return q, q_list