import torch
import torch.nn as nn
from torch import autograd
import torch.nn.utils.spectral_norm as spectral_norm


class DiscriminatorMulti(nn.Module):
    def __init__(
            self, state_dim, amp_reward_coef, hidden_layer_sizes, device,
            num_frames=2, additive_reward_scale=0.02):
        super(DiscriminatorMulti, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.num_frames = num_frames  # 存储帧数参数

        self.amp_reward_coef = amp_reward_coef
        self.additive_reward_scale = additive_reward_scale
        amp_layers = []

        curr_in_dim = state_dim * num_frames
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(spectral_norm(nn.Linear(curr_in_dim, hidden_dim)))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = spectral_norm(nn.Linear(hidden_layer_sizes[-1], 1)).to(device)

        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_states,  # 改为接收多帧状态列表
                         lambda_=10):
        # 将多帧状态沿最后一个维度拼接
        expert_data = expert_states.flatten(1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen


    def get_disc_weights(self):
        weights = []
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.amp_linear.weight))
        return weights

    def get_disc_logit_weights(self):
        return torch.flatten(self.amp_linear.weight)

    def predict_amp_reward(
            self, states,  # 改为接收多帧状态列表
            task_reward, normalizer=None,
            amp_reward_coef=None):
        """
            states: torch.Tensor, shape=(num_envs, num_frames, state_dim)
            task_reward: torch.Tensor, shape=(num_envs, 1)

        The style reward is ADDITIVE: task_reward + coef * scale * [0..1].
        The style score is bounded in [0, 1], so the per-step style ceiling is
        amp_reward_coef * additive_reward_scale -- check that against the
        measured task reward before retuning the coefficient.
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                # 对每一帧状态进行归一化
                states = normalizer.normalize_torch(states, self.device)

            # 拼接多帧状态
            state_cat = states.flatten(1)
            d = self.amp_linear(self.trunk(state_cat))
            reward_coef = self.amp_reward_coef if amp_reward_coef is None else amp_reward_coef
            disc_reward = reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            disc_reward *= self.additive_reward_scale
            reward = task_reward.unsqueeze(-1) + disc_reward
            self.train()
            return reward.squeeze(-1), d, disc_reward.squeeze(-1)
