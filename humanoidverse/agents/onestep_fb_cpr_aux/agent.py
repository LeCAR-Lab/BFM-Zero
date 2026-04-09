# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Dict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

# from ..base import BaseConfig
from ..fb_cpr_aux.agent import FBcprAuxAgent, FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import OneStepFBcprAuxModelConfig


class OneStepFBcprAuxAgentTrainConfig(FBcprAuxAgentTrainConfig):
    pass


class OneStepFBcprAuxAgentConfig(FBcprAuxAgentConfig):
    name: tp.Literal["OneStepFBcprAuxAgent"] = "OneStepFBcprAuxAgent"

    model: OneStepFBcprAuxModelConfig = OneStepFBcprAuxModelConfig()
    train: OneStepFBcprAuxAgentTrainConfig = OneStepFBcprAuxAgentTrainConfig()

    def build(self, obs_space, action_dim: int) -> "OneStepFBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return OneStepFBcprAuxAgent


class OneStepFBcprAuxAgent(FBcprAuxAgent):
    """One-step FB variant of FBcprAuxAgent.

    Replaces the goal-conditioned TD-based FB representation learning (used in
    FBcprAux) with the one-step LSIF objective from: https://arxiv.org/abs/2602.11399 (onestep FB).

    Key differences from FBcprAuxAgent:
    - Backward map B(s) conditions on both obs and action (not only obs/goal).
    - Forward map F(s, a) conditions on obs and action without latent z input.
    - update_fb uses the LSIF one-step loss with B(s) as the target backward.
    - encode_expert and sample_mixed_z use (obs, action) pairs for encoding.
    """

    config_class = OneStepFBcprAuxAgentConfig

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs, train_next_action = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
            train_batch["next"]["action"].to(self.device),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                next_action=train_next_action,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        # compute scalar auxiliary reward as a weighted sum of auxiliary terms
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # keep per-term aux logs for wandb, same as FBcprAuxAgent
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(
                self.device
            )
        aux_reward = self._model._aux_reward_normalizer(aux_reward)
        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        next_action: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        """One-step LSIF forward-backward representation update.

        Unlike the regular TD-based update in FBAgent, this uses the one-step
        LSIF (Least-Squares Importance Fitting) objective where:
        - F(s, a) is the forward representation (no z conditioning).
        - B(s) is the backward representation.
        - Target successor measures use target_F(s', a') and target_B(s').
        """
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                # target forward: F(s', a')
                target_Fs = self._model._target_forward_map(next_obs, next_action)  # num_parallel x batch x z_dim
                # target backward: B(s')  (next obs, not current)
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch
            
            # compute FB representations
            Fs = self._model._forward_map(obs, action)  # num_parallel x batch x z_dim
            B = self._model._backward_map(goal)  # batch x z_dim
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # compute orthonormality loss for backward embedding
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss += self.cfg.train.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_cov = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_cov * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            # compute discriminator reward loss
            Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute auxiliary reward loss
            Qs_aux = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward: F(s, a) · z  (one-step: no z in forward function)
            Fs = self._model._forward_map(obs, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = (
                -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
                - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
                - Q_fb.mean()
            )

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics
