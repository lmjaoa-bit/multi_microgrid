from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from core.reward_v2 import stable_contract_hash
from .guidance import (CounterfactualTeacher, TrainingContext, critic_pair_loss,
                       dr_logit_loss, resolve_guidance)
from .reference import Matd3QReference
from .replay_buffer import CounterfactualReplayBuffer


class Matd3DrCounterfactual(Matd3QReference):
    """MATD3-Q with training-only real-return DR contrasts and output supervision."""

    SLUG = "matd3_dr_counterfactual"
    VERSION = "training_only_closed_loop_dr_contrasts_v2"

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        if not env.reward_v2_enabled:
            raise ValueError("DR counterfactual training requires Reward V2")
        self.guidance = resolve_guidance(config.get("algorithm", config).get("dr_counterfactual", {}))
        self.algorithm_settings = copy.deepcopy(config.get("algorithm", config))
        self.algorithm_settings["name"] = self.SLUG
        self.algorithm_settings["dr_counterfactual"] = copy.deepcopy(self.guidance)
        self.teacher = CounterfactualTeacher(self.agent_ids, gamma=self.gamma,
                                             minimum_gain=self.guidance["minimum_gain"])
        self.teacher_rng = np.random.default_rng(seed + 290001)
        self._training_step = 0
        self._capture_cursor = 0
        self._critic_updates = {mg: 0 for mg in self.agent_ids}
        self._actor_updates = {mg: 0 for mg in self.agent_ids}
        self._last_pair_metrics = {}
        self._diagnostics_directory = None
        self._diagnostic_phase = "training"
        self.replay = CounterfactualReplayBuffer(
            int(config.get("algorithm", config).get("replay_capacity", 200000)),
            self.guidance["capacity"], seed, self._create_labels)
        contract = copy.deepcopy(self.reward_v2_contract)
        contract.pop("signature", None)
        contract["algorithm_slug"] = self.SLUG
        contract["signature"] = stable_contract_hash(contract)
        self.reward_v2_contract = contract

    def capture_training_context(self, env, observations, global_state, metadata):
        self.replay.pending_context = None
        self._training_step = int(metadata.get("total_steps", self._training_step))
        if env.split != "train" or (self.guidance["critic_weight"] == 0
                                    and self.guidance["actor_weight"] == 0):
            return
        probability = self.guidance["terminal_capture_probability"] if (
            env.t == env.episode_length - 1) else self.guidance["capture_probability"]
        if self.teacher_rng.random() >= probability:
            return
        for offset in range(len(self.agent_ids)):
            index = (self._capture_cursor + offset) % len(self.agent_ids)
            agent = self.agent_ids[index]
            low, high = env.dr_devices[agent].feasible_bounds(env.t)
            if high - low > 1e-6:
                self._capture_cursor = (index + 1) % len(self.agent_ids)
                self.replay.pending_context = TrainingContext(
                    copy.deepcopy(env), {mg: obs.copy() for mg, obs in observations.items()},
                    global_state.copy(), agent, self._training_step)
                return

    def _create_labels(self, context, actions):
        return self.teacher.compare(context.env, context.observations, context.state,
                                    actions, context.agent,
                                    lambda obs: self.select_actions(obs, noise_std=0.0),
                                    step=context.step)

    def _sample_labels(self, agent, *, approved_only=False):
        return self.replay.sample_labels(agent, self.guidance["batch_size"],
                                        self._training_step, self.guidance["max_age_steps"],
                                        approved_only=approved_only)

    def _tensor(self, values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)

    def _fit_counterfactual_critics(self):
        if self.guidance["critic_weight"] == 0:
            return {}
        metrics = {}
        for agent in self.agent_ids:
            labels = self._sample_labels(agent)
            if not labels:
                continue
            states = self._tensor([label.state for label in labels])
            reference, _ = self.project_actions(states, self._tensor([label.reference_actions for label in labels]))
            candidates, _ = self.project_actions(states, self._tensor([label.candidate_actions for label in labels]))
            critic = self.critics[agent]
            q_reference = critic(states, reference.flatten(1))
            q_candidates = critic(states, candidates.flatten(1))
            gains = self._tensor([[label.gain] for label in labels])
            pair_loss = critic_pair_loss(q_reference, q_candidates, gains)
            terminal = self._tensor([label.rollout_steps == 1 for label in labels]).bool()
            anchor = pair_loss * 0.0
            if terminal.any():
                true_reference = self._tensor([[label.reference_return] for label in labels])[terminal]
                true_candidate = self._tensor([[label.candidate_return] for label in labels])[terminal]
                anchor = sum(torch.nn.functional.smooth_l1_loss(q[terminal], target)
                             for values, target in ((q_reference, true_reference), (q_candidates, true_candidate))
                             for q in values)
            loss = self.guidance["critic_weight"] * (
                pair_loss + self.guidance["terminal_anchor_weight"] * anchor)
            optimizer = self.critic_optimizers[agent]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            self._critic_updates[agent] += 1
            errors = [((candidate.detach() - base.detach()) * gains < 0).float().mean()
                      for base, candidate in zip(q_reference, q_candidates)]
            metrics[agent] = dict(pair_loss=float(pair_loss.detach()),
                                  ordering_error_rate=float(torch.stack(errors).mean()),
                                  sampled_labels=len(labels))
        self._last_pair_metrics = metrics
        return metrics

    def update(self, batch_size):
        if len(self.replay) < max(int(batch_size), self.learning_starts):
            return {}
        pair_metrics = {}
        if (self.update_steps + 1) % self.guidance["update_interval"] == 0:
            pair_metrics = self._fit_counterfactual_critics()
        metrics = super().update(batch_size)
        if metrics and pair_metrics:
            metrics["dr_counterfactual"] = pair_metrics
        return metrics

    def actor_auxiliary_loss(self, active, state, joint_raw, observations):
        loss, metrics = super().actor_auxiliary_loss(active, state, joint_raw, observations)
        if self.guidance["actor_weight"] == 0:
            return loss, metrics
        labels = self._sample_labels(active, approved_only=True)
        if not labels:
            return loss, metrics
        teacher_obs = self._tensor([label.observation for label in labels])
        logits = self.actors[active].network[:-1](teacher_obs)
        targets = self._tensor([label.candidate_raw for label in labels])
        auxiliary = dr_logit_loss(logits, targets, weight=self.guidance["actor_weight"])
        self._actor_updates[active] += 1
        metrics.update(dr_counterfactual_actor_loss=auxiliary.detach(),
                       dr_counterfactual_gain=self._tensor([label.gain for label in labels]).mean(),
                       dr_counterfactual_approved_labels=logits.new_tensor(len(labels)))
        return loss + auxiliary, metrics

    def save(self, path):
        payload = dict(algorithm_slug=self.SLUG, action_semantics=self.projector.semantics,
                       projection_signature=self.projection_signature,
                       observation_schemas=self.observation_schemas, state_schema=self.state_schema,
                       reward_v2_contract=copy.deepcopy(self.reward_v2_contract),
                       dr_counterfactual=copy.deepcopy(self.guidance),
                       algorithm_settings=copy.deepcopy(self.algorithm_settings),
                       dr_counterfactual_version=self.VERSION, update_steps=self.update_steps)
        payload.update({name: getattr(self, name).state_dict()
                        for name in ("actors", "target_actors", "critics", "target_critics")})
        torch.save(payload, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if (checkpoint.get("dr_counterfactual") != self.guidance
                or checkpoint.get("dr_counterfactual_version") != self.VERSION):
            raise ValueError("Incompatible DR counterfactual training configuration")
        if checkpoint.get("algorithm_settings") != self.algorithm_settings:
            raise ValueError("Incompatible DR algorithm configuration")
        super().load(path)

    def configure_diagnostics(self, directory, metadata):
        self._diagnostics_directory = Path(directory)
        self._diagnostics_directory.mkdir(parents=True, exist_ok=True)

    def set_diagnostic_phase(self, phase, metadata):
        if self._diagnostics_directory is None:
            return
        if phase is not None:
            if phase not in {"training", "validation", "formal_test"}:
                raise ValueError("Unknown DR diagnostic phase")
            self._diagnostic_phase = phase
        elif metadata.get("training_complete"):
            self._diagnostic_phase = "training"
        summary = dict(version=self.VERSION, inference_uses_teacher=False,
                       label_source_split="train", compare_calls=self.teacher.compare_calls,
                       simulated_suffix_steps=self.teacher.rollout_steps,
                       retained_labels={mg: len(self.replay.by_agent.get(mg, ())) for mg in self.agent_ids},
                       critic_updates=self._critic_updates, actor_updates=self._actor_updates,
                       last_pair_metrics=self._last_pair_metrics,
                       phase=phase, phase_metadata=metadata)
        name = f"dr_counterfactual_{self._diagnostic_phase}_summary.json"
        (self._diagnostics_directory / name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
