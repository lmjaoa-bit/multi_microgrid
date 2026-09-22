from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.base import RecoverableAlgorithmProtocol
from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
    Matd3HierarchicalTransformerPredictiveDr,
)
from algorithms.runtime_helpers import FastReplayBuffer
from algorithms.training_recovery import (
    _atomic_torch_save,
    _atomic_write_json,
    read_transition_chunk,
    sha256_file,
    validate_fingerprints,
    write_transition_chunk,
)
from envs.multi_microgrid_env import MultiMicrogridEnv


AGENTS = ["mg1", "mg2", "mg3"]


def _add_numbered_transition(replay: FastReplayBuffer, number: int) -> None:
    observations = {mg: np.array([number, i], dtype=np.float32) for i, mg in enumerate(AGENTS)}
    actions = {mg: np.array([number, -number], dtype=np.float32) for mg in AGENTS}
    replay.add(
        observations,
        np.array([number], dtype=np.float32),
        actions,
        float(number),
        {mg: value + 1 for mg, value in observations.items()},
        np.array([number + 1], dtype=np.float32),
        number % 4 == 3,
    )


def test_fast_replay_restore_preserves_ring_order_position_and_rng() -> None:
    original = FastReplayBuffer(capacity=5, seed=17)
    for number in range(8):
        _add_numbered_transition(original, number)

    logical = original.logical_transitions()
    metadata = original.state_metadata()
    assert [int(row.global_state[0]) for row in logical] == [3, 4, 5, 6, 7]

    restored = FastReplayBuffer(capacity=5, seed=999)
    restored.restore_logical(logical, metadata)

    assert restored._next_index == original._next_index
    assert restored._total_added == 8
    assert [int(row.global_state[0]) for row in restored.logical_transitions()] == [3, 4, 5, 6, 7]
    for _ in range(3):
        left = original.sample(3, AGENTS, torch.device("cpu"))["global_state"]
        right = restored.sample(3, AGENTS, torch.device("cpu"))["global_state"]
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    "metadata_patch,match",
    [
        ({"capacity": 6}, "capacity"),
        ({"size": 4}, "size"),
        ({"next_index": 5}, "next_index"),
        ({"total_added": 4}, "total_added"),
    ],
)
def test_fast_replay_restore_rejects_incompatible_metadata(metadata_patch, match) -> None:
    replay = FastReplayBuffer(capacity=5, seed=1)
    for number in range(5):
        _add_numbered_transition(replay, number)
    metadata = replay.state_metadata()
    metadata.update(metadata_patch)
    with pytest.raises(ValueError, match=match):
        FastReplayBuffer(capacity=5).restore_logical(replay.logical_transitions(), metadata)


def test_optional_recovery_protocol_does_not_change_base_algorithm_contract() -> None:
    class Recoverable:
        def save_training_state(self, directory: Path, progress):
            return {"directory": str(directory), "progress": dict(progress)}

        def load_training_state(self, directory: Path):
            return {"directory": str(directory)}

    class Ordinary:
        pass

    assert isinstance(Recoverable(), RecoverableAlgorithmProtocol)
    assert not isinstance(Ordinary(), RecoverableAlgorithmProtocol)


def test_atomic_writers_and_transition_chunk_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "state.json"
    tensor_path = tmp_path / "state.pt"
    _atomic_write_json({"episode": 7}, json_path)
    _atomic_torch_save({"weight": torch.tensor([2.0])}, tensor_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"episode": 7}
    assert torch.equal(torch.load(tensor_path, weights_only=False)["weight"], torch.tensor([2.0]))
    assert len(sha256_file(tensor_path)) == 64
    assert not list(tmp_path.glob("*.tmp"))

    replay = FastReplayBuffer(capacity=5, seed=3)
    for number in range(3):
        _add_numbered_transition(replay, number)
    chunk = tmp_path / "chunk.pt"
    digest = write_transition_chunk(chunk, replay.logical_transitions())
    restored = read_transition_chunk(chunk, expected_sha256=digest)
    assert [int(row.global_state[0]) for row in restored] == [0, 1, 2]


def test_fingerprint_validation_names_the_mismatch() -> None:
    expected = {"algorithm_slug": "peak", "schema": ["a", "b"]}
    validate_fingerprints(expected, dict(expected))
    with pytest.raises(ValueError, match="schema"):
        validate_fingerprints(expected, {"algorithm_slug": "peak", "schema": ["a"]})


def _predictive_algorithm(seed: int = 42):
    env_config = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(
        Path("configs/algorithms/matd3_hierarchical_transformer_predictive_dr_hvac.yaml")
        .read_text(encoding="utf-8")
    )
    env = MultiMicrogridEnv(env_config, split="train")
    algorithm = Matd3HierarchicalTransformerPredictiveDr.from_env(
        env, algorithm_config, device="cpu", seed=seed
    )
    return env, algorithm


def _populate_predictive_replay(env, algorithm, count: int = 12) -> None:
    observations, reset_info = env.reset(seed=321)
    state = reset_info["global_state"]
    for index in range(count):
        actions = algorithm.select_actions(observations, noise_std=0.05)
        next_observations, rewards, done, info = env.step(actions)
        algorithm.replay.add(
            observations,
            state,
            actions,
            float(sum(rewards.values())),
            next_observations,
            info["global_state_after"],
            done,
        )
        if done and index + 1 < count:
            observations, reset_info = env.reset(seed=322 + index)
            state = reset_info["global_state"]
        else:
            observations, state = next_observations, info["global_state_after"]


def _assert_module_state_equal(left, right) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key]), key


def test_predictive_training_state_resume_matches_uninterrupted_updates(tmp_path: Path) -> None:
    env, source = _predictive_algorithm()
    _populate_predictive_replay(env, source)
    initial = tmp_path / "initial"
    source.save_training_state(initial, {"completed_episode": 0})

    _, continuous = _predictive_algorithm(seed=99)
    continuous.load_training_state(initial)
    for _ in range(6):
        continuous.update(batch_size=4)

    _, interrupted = _predictive_algorithm(seed=100)
    interrupted.load_training_state(initial)
    for _ in range(3):
        interrupted.update(batch_size=4)
    resume_dir = tmp_path / "resume"
    interrupted.save_training_state(resume_dir, {"completed_episode": 3})

    _, resumed = _predictive_algorithm(seed=101)
    progress = resumed.load_training_state(resume_dir)
    assert progress["completed_episode"] == 3
    for _ in range(3):
        resumed.update(batch_size=4)

    assert resumed.update_steps == continuous.update_steps
    _assert_module_state_equal(resumed.actors, continuous.actors)
    _assert_module_state_equal(resumed.critics, continuous.critics)
    left_batch = continuous.replay.sample(4, continuous.agent_ids, torch.device("cpu"))
    right_batch = resumed.replay.sample(4, resumed.agent_ids, torch.device("cpu"))
    assert torch.equal(left_batch["global_state"], right_batch["global_state"])
    observations, _ = env.reset(seed=777)
    left_actions = continuous.select_actions(observations, noise_std=0.1)
    right_actions = resumed.select_actions(observations, noise_std=0.1)
    for mg in continuous.agent_ids:
        assert np.array_equal(left_actions[mg], right_actions[mg])


@pytest.mark.parametrize(
    "fingerprint_key",
    [
        "algorithm_slug",
        "config_signature",
        "state_schema",
        "projection_signature",
        "normalization_signature",
    ],
)
def test_predictive_training_state_rejects_fingerprint_before_mutation(
    tmp_path: Path, fingerprint_key: str
) -> None:
    env, source = _predictive_algorithm()
    _populate_predictive_replay(env, source, count=4)
    directory = tmp_path / fingerprint_key
    source.save_training_state(directory, {"completed_episode": 2})
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprints"][fingerprint_key] = "incompatible"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, target = _predictive_algorithm(seed=98)
    before = {name: value.detach().clone() for name, value in target.actors.state_dict().items()}
    with pytest.raises(ValueError, match=fingerprint_key):
        target.load_training_state(directory)
    for name, value in target.actors.state_dict().items():
        assert torch.equal(value, before[name])


def test_training_state_ignores_uncommitted_orphan_chunk(tmp_path: Path) -> None:
    env, source = _predictive_algorithm()
    _populate_predictive_replay(env, source, count=4)
    directory = tmp_path / "recovery"
    source.save_training_state(directory, {"completed_episode": 4})
    write_transition_chunk(directory / "replay-orphan.pt", source.replay.logical_transitions())

    _, restored = _predictive_algorithm(seed=97)
    progress = restored.load_training_state(directory)
    assert progress == {"completed_episode": 4}
    assert len(restored.replay) == 4
