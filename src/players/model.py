import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from players.base_player import BasePlayer


class MultiDiscreteActionMaskModel(TorchModelV2, nn.Module, BasePlayer):
    """TorchModelV2 for Dict(obs, action_mask) + MultiDiscrete actions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        if (
            hasattr(original_space, "spaces")
            and "observations" in original_space.spaces
        ):
            self._obs_shape = original_space["observations"].shape
        else:
            self._obs_shape = obs_space.shape

        in_channels = int(self._obs_shape[0])

        self.encoder = nn.Sequential(
            # no dilation since first layers must detect borders
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # dilation to look at broader territory
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *self._obs_shape, dtype=torch.float32)
            flat_size = self.encoder(dummy).shape[1]
            # print("Input size for the trunk:", flat_size)
        stat_size = int(original_space["stats"].shape[0])

        self.trunk = nn.Sequential(
            nn.Linear(flat_size + stat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        restored = restore_original_dimensions(
            input_dict["obs"], self.obs_space, "torch"
        )
        obs = restored["observations"].float()
        action_mask = restored["action_mask"].float()
        stats = restored["stats"].float()

        obs_trunked = self.encoder(obs)

        combined_features = torch.cat([obs_trunked, stats], dim=1)

        features = self.trunk(combined_features)
        logits = self.policy_head(features)

        # MultiDiscrete logits are flattened as [target_logits..., commit_logits...].
        # Our action_mask follows the same layout.
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e20)
        masked_logits = logits + inf_mask

        self._value_out = self.value_head(features).squeeze(-1)
        return masked_logits, state

    def value_function(self):
        return self._value_out


MODEL_NAME = "multi_discrete_action_mask_model"
ModelCatalog.register_custom_model(MODEL_NAME, MultiDiscreteActionMaskModel)
