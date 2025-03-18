import pathlib

# Use dot here to denote importing the file in the folder hosting this file.
import os.path as osp
import torch

from .common.utils import load_config_and_model
from .common.net import Net, ActorProb, ActorCritic, DoubleCritic
from .common.lagrangian import LagrangianPolicy

from torch.distributions import Independent, Normal
import numpy as np

FOLDER_ROOT = pathlib.Path(__file__).parent  # The path to the folder hosting this file.

class Policy:
    """
    This class is the interface where the evaluation scripts communicate with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    Do not change the name of this class.

    Please do not import any external package.
    """
    # FILLED YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "Shawn"  # Your preferred name here in a string
    CREATOR_UID = "h-shawn"  # Your UID here in a string

    def __init__(self):
        cfg, model = load_config_and_model(FOLDER_ROOT, True)

        state_shape = 259
        action_shape = 2
        max_action = 1.0
        cost_dim = 1
        hidden_sizes = cfg["hidden_sizes"]
        unbounded = cfg["unbounded"]
        last_layer_scale = cfg["last_layer_scale"]
        device='cpu'

        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)

        actor = ActorProb(
            net,
            action_shape,
            max_action=max_action,
            device=device,
            conditioned_sigma=True,
            unbounded=unbounded
        ).to(device)

        critics = []
        for _ in range(1 + cost_dim):
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            critics.append(DoubleCritic(net1, net2, device=device).to(device))

        actor_critic = ActorCritic(actor, critics)

        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        if last_layer_scale:
            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        self.policy = LagrangianPolicy(
            actor,
            critics,
            None,
        )

        self.policy.load_state_dict(model["model"])

    def __call__(self, obs):
        with torch.no_grad():
            logits, hidden = self.policy.actor(obs)
            assert isinstance(logits, tuple)
            dist = Independent(Normal(*logits), 1)

            if self.policy._deterministic_eval and not self.policy.training:
                act = logits[0]
            else:
                act = dist.rsample()
            act = torch.tanh(act)
                
        act = act.detach().cpu().numpy()
        action = np.squeeze(self.policy.map_action(act))
        return action
