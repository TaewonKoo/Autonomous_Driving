"""Categorical/Gaussian PPO, KL-penalty PPO, TRPO and parameterized SAC."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal, kl_divergence
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils.networks import build_mlp


class Policy(nn.Module):
    def __init__(self, profile, env):
        super().__init__()
        self.discrete = env.discrete
        self.recurrent = profile.network == "gru"
        width = env.discrete or env.action_dim
        if self.recurrent:
            self.gru = nn.GRU(env.obs_dim, 5, batch_first=True)
            self.actor = nn.Linear(5, width)
        else:
            self.actor = build_mlp(env.obs_dim, profile.actor, width, profile.activation)
        self.critic = None if profile.algorithm == "trpo" else build_mlp(
            env.obs_dim, profile.critic, 1, profile.activation)
        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((width,), -.5))

    def distribution(self, state, hidden=None):
        if self.recurrent:
            if state.ndim == 2:
                state = state.unsqueeze(0)
                state, hidden = self.gru(state, hidden)
                state = state.squeeze(0)
            else:
                state, hidden = self.gru(state, hidden)
        output = self.actor(state)
        if self.discrete:
            return Categorical(logits=output), hidden
        return Independent(Normal(output, self.log_std.clamp(-5, 2).exp()), 1), hidden

    def value(self, state):
        if self.critic is None:
            return torch.zeros(state.shape[:-1], device=state.device)
        return self.critic(state).squeeze(-1)


def discounted(rewards, end_value, gamma, continuations=None):
    result = np.empty_like(rewards)
    for i in reversed(range(len(rewards))):
        keep = 1. if continuations is None else continuations[i]
        end_value = rewards[i] + gamma * keep * end_value
        result[i] = end_value
    return result


def conjugate_gradient(product, rhs, iterations=10):
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    square = residual.dot(residual)
    for _ in range(iterations):
        if square < 1e-12:
            break
        projected = product(direction)
        distance = square / projected.dot(direction).clamp_min(1e-12)
        solution += distance * direction
        residual -= distance * projected
        next_square = residual.dot(residual)
        direction = residual + next_square / square * direction
        square = next_square
    return solution


class OnPolicy:
    def __init__(self, profile, env, device):
        self.p = profile
        self.device = device
        self.model = Policy(profile, env).to(device)
        self.beta = 1.
        if profile.algorithm != "trpo":
            self.actor_optimizer = torch.optim.Adam(
                [p for name, p in self.model.named_parameters() if not name.startswith("critic.")],
                lr=profile.lr)
            self.value_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=profile.lr)

    def tensor(self, array):
        return torch.as_tensor(array, device=self.device)

    @torch.no_grad()
    def action(self, state, hidden=None, deterministic=False):
        dist, hidden = self.model.distribution(self.tensor(state), hidden)
        if deterministic:
            action = dist.probs.argmax(-1) if self.model.discrete else dist.mean
        else:
            action = dist.sample()
        return action.cpu().numpy(), hidden

    def update(self, trajectories, progress):
        p = self.p
        observations, actions, advantages, returns = [], [], [], []
        for obs, act, rewards, next_obs, active, continuations in trajectories:
            if p.algorithm == "trpo":
                returns.append(discounted(rewards, 0., p.gamma).reshape(-1))
            else:
                with torch.no_grad():
                    value = self.model.value(self.tensor(obs)).cpu().numpy()
                    next_value = self.model.value(self.tensor(next_obs)).cpu().numpy()
                next_values = np.concatenate((value[1:], next_value[None, :]))
                deltas = rewards + p.gamma * continuations * next_values - value
                advantage = discounted(deltas, 0., p.gamma * p.gae, continuations)
                advantages.append(advantage[active])
                returns.append((advantage + value)[active])
            observations.append(obs[active])
            actions.append(act[active])
        x, a = self.tensor(np.concatenate(observations)), self.tensor(np.concatenate(actions))
        target = self.tensor(np.concatenate(returns)).float()
        if p.algorithm == "trpo":
            if self.model.recurrent and len(trajectories) != 1:
                raise ValueError("The recurrent Flow policy requires one complete sequence per update.")
            t = torch.cat([torch.linspace(0, 1, len(obs), device=self.device) for obs in observations])
            features = torch.column_stack((x, x.square(), t, t.square(), t**3, torch.ones_like(t)))
            gram = features.T @ features + torch.eye(features.shape[1], device=self.device) * 1e-4
            advantage = target - features @ torch.linalg.solve(gram, features.T @ target)
        else:
            advantage = self.tensor(np.concatenate(advantages)).float()
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        with torch.no_grad():
            old, _ = self.model.distribution(x)
            old_log_prob = old.log_prob(a)
        if p.algorithm == "trpo":
            return self.trpo(x, a, advantage, old, old_log_prob)
        for optimizer in (self.actor_optimizer, self.value_optimizer):
            optimizer.param_groups[0]["lr"] = p.lr * max(0., 1. - progress) if p.anneal else p.lr
        for _ in range(p.epochs):
            for index in torch.randperm(len(x), device=self.device).split(p.batch):
                new, _ = self.model.distribution(x[index])
                ratio = (new.log_prob(a[index]) - old_log_prob[index]).exp()
                objective = ratio * advantage[index]
                if p.algorithm == "ppo":
                    objective = torch.minimum(objective, ratio.clamp(1 - p.clip, 1 + p.clip)
                                               * advantage[index])
                else:
                    prior = Independent(Normal(old.base_dist.loc[index], old.base_dist.scale[index]), 1)
                    objective = objective - self.beta * kl_divergence(prior, new)
                self.actor_optimizer.zero_grad(set_to_none=True)
                (-objective.mean()).backward()
                self.actor_optimizer.step()
                self.value_optimizer.zero_grad(set_to_none=True)
                F.mse_loss(self.model.value(x[index]), target[index]).backward()
                self.value_optimizer.step()
        with torch.no_grad():
            observed_kl = float(kl_divergence(old, self.model.distribution(x)[0]).mean())
        if p.algorithm == "kl_ppo":
            if observed_kl > 1.5 * p.kl:
                self.beta *= 2
            elif observed_kl < p.kl / 1.5:
                self.beta *= .5
            self.beta = min(1e4, max(1e-4, self.beta))
        return {"kl": observed_kl, "beta": self.beta if p.algorithm == "kl_ppo" else None}

    def trpo(self, state, action, advantage, old, old_log_prob):
        params = tuple(self.model.parameters())

        def objective():
            new, _ = self.model.distribution(state)
            return ((new.log_prob(action) - old_log_prob).exp() * advantage).mean()

        def divergence():
            return kl_divergence(old, self.model.distribution(state)[0]).mean()

        def hvp(vector):
            if self.model.recurrent:
                # Finite differences of the exact KL gradient avoid a second
                # derivative through the recurrent backend.
                epsilon = 1e-3 / vector.norm().clamp_min(1e-6)
                with torch.no_grad():
                    vector_to_parameters(original + epsilon * vector, params)
                positive = parameters_to_vector(torch.autograd.grad(divergence(), params))
                with torch.no_grad():
                    vector_to_parameters(original - epsilon * vector, params)
                negative = parameters_to_vector(torch.autograd.grad(divergence(), params))
                with torch.no_grad():
                    vector_to_parameters(original.clone(), params)
                return (positive - negative) / (2 * epsilon) + .1 * vector
            first = parameters_to_vector(torch.autograd.grad(divergence(), params, create_graph=True))
            second = parameters_to_vector(torch.autograd.grad(first.dot(vector), params))
            return second + .1 * vector

        original = parameters_to_vector(params).detach()
        score = objective()
        gradient = parameters_to_vector(torch.autograd.grad(score, params)).detach()
        direction = conjugate_gradient(hvp, gradient)
        curvature = direction.dot(hvp(direction))
        if not torch.isfinite(curvature) or curvature <= 0:
            return {"kl": 0., "accepted": False}
        direction *= torch.sqrt(2 * self.p.kl / curvature)
        with torch.no_grad():
            for exponent in range(10):
                vector_to_parameters(original + .5**exponent * direction, params)
                kl = float(divergence())
                if 0 <= kl <= self.p.kl and objective() > score:
                    return {"kl": kl, "accepted": True}
            vector_to_parameters(original, params)
        return {"kl": 0., "accepted": False}


class SACPolicy(nn.Module):
    def __init__(self, profile, obs_dim, hybrid):
        super().__init__()
        self.hybrid = hybrid
        self.net = build_mlp(obs_dim, profile.actor, 6 if hybrid else 2, profile.activation)

    def forward(self, state, deterministic=False):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        normal = Normal(mean, log_std.clamp(-8., 2.).exp())
        raw = mean if deterministic else normal.rsample()
        squashed = raw.tanh()
        log_prob = (normal.log_prob(raw) - 2 * (math.log(2.) - raw - F.softplus(-2 * raw))).sum(-1)
        if not self.hybrid:
            return squashed, log_prob
        scores = squashed[..., 1:]
        soft = scores.softmax(dim=-1)
        hard = F.one_hot(scores.argmax(dim=-1), 2).to(scores.dtype)
        return torch.cat((squashed[..., :1], hard + (soft - soft.detach())), dim=-1), log_prob


class SAC:
    def __init__(self, profile, env, device, seed):
        self.p = profile
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.hybrid = profile.algorithm == "pasac"
        self.model = SACPolicy(profile, env.obs_dim, self.hybrid).to(device)
        self.action_dim = 3 if self.hybrid else 1
        self.q = nn.ModuleList(build_mlp(env.obs_dim + self.action_dim, profile.critic, 1,
                                         profile.activation) for _ in range(2)).to(device)
        self.target = copy.deepcopy(self.q).requires_grad_(False)
        self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=profile.lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=profile.lr)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.temperature_optimizer = torch.optim.Adam([self.log_alpha], lr=profile.lr)
        self.target_entropy = -float(self.action_dim)
        self.capacity = min(200_000, profile.steps)
        self.replay = np.empty((self.capacity, 2 * env.obs_dim + self.action_dim + 2), np.float32)
        self.obs_dim = env.obs_dim
        self.count = 0

    def tensor(self, array):
        return torch.as_tensor(array, device=self.device)

    @torch.no_grad()
    def action(self, state, hidden=None, deterministic=False):
        encoded, _ = self.model(self.tensor(state), deterministic)
        if self.hybrid:
            encoded = torch.cat((encoded[:, :1], encoded[:, 1:].argmax(-1, keepdim=True)), dim=1)
        return encoded.cpu().numpy(), None

    def sample_random(self):
        values = [self.rng.uniform(-1, 1)]
        if self.hybrid:
            values.append(self.rng.integers(2))
        return np.asarray([values], np.float32)

    def add(self, state, action, reward, next_state, terminated):
        action = action[0]
        if self.hybrid:
            action = [action[0], int(action[1] == 0), int(action[1] == 1)]
        self.replay[self.count % self.capacity] = np.concatenate((
            state[0], action, [reward[0], terminated], next_state[0]))
        self.count += 1

    def update(self):
        sample = self.tensor(self.replay[self.rng.integers(0, min(self.capacity, self.count), self.p.batch)])
        x, a, r, done, next_x = sample.split((self.obs_dim, self.action_dim, 1, 1, self.obs_dim), dim=1)
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_a, next_log_prob = self.model(next_x)
            next_sa = torch.cat((next_x, next_a), dim=1)
            target = r + self.p.gamma * (1 - done) * (
                torch.minimum(self.target[0](next_sa), self.target[1](next_sa)) -
                alpha * next_log_prob[:, None])
        sa = torch.cat((x, a), dim=1)
        q_loss = sum(F.mse_loss(q(sa), target) for q in self.q)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.q.requires_grad_(False)
        policy_action, log_prob = self.model(x)
        sa = torch.cat((x, policy_action), dim=1)
        actor_loss = (alpha * log_prob - torch.minimum(self.q[0](sa), self.q[1](sa)).squeeze(1)).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy_optimizer.step()
        self.q.requires_grad_(True)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean().backward()
        self.temperature_optimizer.step()
        with torch.no_grad():
            for source, target_param in zip(self.q.parameters(), self.target.parameters()):
                target_param.lerp_(source, .005)
        return (q_loss.detach(), actor_loss.detach())
