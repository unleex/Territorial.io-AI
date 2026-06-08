import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from players.base_player import BasePlayer


class MultiDiscreteActionMaskModel(RecurrentNetwork, nn.Module, BasePlayer):
    """TorchModelV2 for Dict(obs, action_mask) + MultiDiscrete actions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        RecurrentNetwork.__init__(
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dilation to look at broader territory
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self._value_out = None

    def get_initial_state(self):
        return [torch.zeros(256), torch.zeros(256)]

    def forward_rnn(self, input_dict, state, seq_lens):
        B, T, flat_dim = input_dict.shape
        flat_inputs = input_dict.view(B * T, flat_dim)

        restored = restore_original_dimensions(flat_inputs, self.obs_space, "torch")
        obs = restored["observations"].float()
        action_mask = restored["action_mask"].float()
        stats = restored["stats"].float()

        obs_trunked = self.encoder(obs)

        combined_features = torch.cat([obs_trunked, stats], dim=1)

        features = self.trunk(combined_features)

        lstm_in = features.view(B, T, 256)

        h = state[0].unsqueeze(0).contiguous()
        c = state[1].unsqueeze(0).contiguous()

        lstm_out, [h_new, c_new] = self.lstm(lstm_in, (h, c))

        flat_lstm_out = lstm_out.reshape(B * T, 256)
        logits = self.policy_head(flat_lstm_out)

        # MultiDiscrete logits are flattened as [target_logits..., commit_logits...].
        # Our action_mask follows the same layout.

        inf_mask = torch.clamp(torch.log(action_mask), min=-1e20)
        masked_logits = logits + inf_mask

        self._value_out = self.value_head(features).squeeze(-1)
        state_out = [h_new.squeeze(0), c_new.squeeze(0)]
        return masked_logits, state_out

    def value_function(self):
        return self._value_out


MODEL_NAME = "multi_discrete_action_mask_model"
ModelCatalog.register_custom_model(MODEL_NAME, MultiDiscreteActionMaskModel)
