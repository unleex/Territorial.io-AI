import torch
import torch.nn as nn
from players.base_player import BasePlayer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

LSTM_HIDDEN_SIZE = 128


class LSTMModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        RecurrentNetwork.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        self.target_dim = int(original_space["action_mask"].shape[0])

        self.obs_shape = original_space["observations"].shape
        self.stat_size = int(original_space["stats"].shape[0])
        self.feature_dim = 256  # output of trunk before masking
        self.time_major = bool(model_config.get("_time_major", False))

        in_channels = int(self.obs_shape[0])

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
            dummy = torch.zeros(1, *self.obs_shape, dtype=torch.float32)
            flat_size = self.encoder(dummy).shape[1]
            # print("Input size for the trunk:", flat_size)

        self.trunk = nn.Sequential(
            nn.Linear(flat_size + self.stat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            batch_first=True,
        )

        self.policy_head = nn.Linear(LSTM_HIDDEN_SIZE, num_outputs)
        self.value_head = nn.Linear(LSTM_HIDDEN_SIZE, 1)
        self._value_out = None

    def get_initial_state(self):
        return [torch.zeros(LSTM_HIDDEN_SIZE), torch.zeros(LSTM_HIDDEN_SIZE)]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observations"].float()
        stats = input_dict["obs"]["stats"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        x = self.encoder(obs)
        x = torch.cat([x, stats], dim=1)
        x = self.trunk(x)

        combined = torch.cat([x, action_mask], dim=1)

        return super().forward({"obs_flat": combined}, state, seq_lens)

    def forward_rnn(self, inputs, state, seq_lens):
        if self.time_major:
            inputs = inputs.transpose(0, 1)

        features = inputs[..., : self.feature_dim]
        action_mask = inputs[..., self.feature_dim :]

        h_in = state[0].unsqueeze(0).contiguous()
        c_in = state[1].unsqueeze(0).contiguous()

        lstm_out, (h_out, c_out) = self.lstm(features, (h_in, c_in))

        flat_out = lstm_out.reshape(-1, LSTM_HIDDEN_SIZE)
        logits = self.policy_head(flat_out)

        mask = action_mask.reshape(-1, logits.shape[-1])
        inf_mask = torch.clamp(torch.log(mask), min=-1e20)
        masked_logits = logits + inf_mask

        self._value_out = self.value_head(flat_out).squeeze(-1)

        if self.time_major:
            masked_logits = masked_logits.view(
                lstm_out.shape[0], lstm_out.shape[1], -1
            ).transpose(0, 1)
        else:
            masked_logits = masked_logits.view(lstm_out.shape[0], lstm_out.shape[1], -1)

        return masked_logits, [h_out.squeeze(0), c_out.squeeze(0)]

    def value_function(self):
        return self._value_out


class ActionMaskModel(TorchModelV2, nn.Module, BasePlayer):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        self.target_dim = int(original_space["action_mask"].shape[0])
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
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Policy head now only outputs target_dim + 1 (just the discrete logits and 1 continuous mean)
        self.policy_head = nn.Linear(512, self.target_dim + 1)
        self.value_head = nn.Linear(512, 1)
        self._value_out = None

        # State-independent trainable parameter for continuous exploration variance
        # Initialized to 0.0 (std = 1.0). If you want less initial exploration, set to -0.5
        self.log_std = nn.Parameter(torch.zeros(1))

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

        # Extract components
        target_logits = logits[..., : self.target_dim]

        # 1. Squash the mean to [0.0, 1.0] using Sigmoid to perfectly match your Box boundaries
        commit_mu = torch.sigmoid(logits[..., self.target_dim])

        # 2. Broadcast the standalone log_std parameter to match the current batch size
        batch_size = logits.shape[0]
        commit_log_std = self.log_std.expand(batch_size)

        # 3. Pack continuous parameters together [batch, 2]
        commit_params = torch.stack([commit_mu, commit_log_std], dim=-1)

        # 4. Apply action masking to target selection
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e20)
        masked_target_logits = target_logits + inf_mask

        # 5. Maintain the precise flattening order expected by RLlib
        masked_logits = torch.cat([commit_params, masked_target_logits], dim=-1)

        self._value_out = self.value_head(features).squeeze(-1)
        return masked_logits, state

    def value_function(self):
        return self._value_out


MODEL_NAME = "action_mask_model"
ModelCatalog.register_custom_model(MODEL_NAME, ActionMaskModel)
