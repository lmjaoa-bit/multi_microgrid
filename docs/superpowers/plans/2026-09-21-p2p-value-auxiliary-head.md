# P2P Value Auxiliary Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated predictive-DR MATD3 Transformer variant whose shared actor representation learns whether an ESS P2P offer will clear and what fraction will clear, without changing the action space, environment, rewards, projector, or canonical MATD3.

**Architecture:** `matd3_hierarchical_transformer_p2p_value` subclasses the corrected predictive-DR algorithm contract `1.1`, replaces only its actors and actor-update implementation, and keeps all physical mappings inherited. Each child actor exposes attended ESS/supply/trade/peer representations to a dual P2P head; labels are reconstructed in physical kWh from replayed next observations, and a masked auxiliary loss is added only during delayed actor updates.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyYAML, pytest, the repository algorithm registry, shared experiment pipeline, and saved-result JSON/CSV gate tooling.

**Spec:** `docs/superpowers/specs/2026-09-21-p2p-value-auxiliary-head-design.md`

## Global Constraints

- Work on branch `codex/milp-benchmark`; do not create a separate worktree unless the user changes the execution request.
- Do not execute this plan in the current task. If the user later authorizes implementation, use native in-session execution without subagents, as previously requested.
- Do not modify `algorithms/matd3/`, the corrected parent package `algorithms/matd3_hierarchical_transformer_predictive_dr/`, shared environment physics, rewards, settlement, evaluation, or plotting.
- The new slug is exactly `matd3_hierarchical_transformer_p2p_value`; its result root is `outputs/working_runs/matd3_hierarchical_transformer_p2p_value/runs/`.
- The first implementation accepts exactly parent predictive contract `1.1` and uses P2P auxiliary contract `1.0`.
- Preserve the two-dimensional `[ESS, DR]` action interface and exact inherited predictive-HVAC, MG2/MG3 feasible mapping, latent exploration, and target smoothing behavior.
- The auxiliary head is training-only; its outputs never enter `select_actions`, the environment, the action projector, target-Q construction, or critic inputs.
- Restore P2P outcomes to physical kWh with the positive per-agent `trade` normalizer before thresholds, ratios, or consistency checks.
- Only offered rows train the auxiliary objective; a no-offer batch must produce a finite differentiable zero.
- First-run auxiliary defaults are hidden size `64`, maximum weight `0.05`, linear ramp `5000` updates, component weights `1.0/1.0`, epsilon `1e-6 kWh`, and consistency tolerance `1e-5 kWh`.
- Formal-test results may be read only after training and checkpoint selection; they may not tune model weights or select the checkpoint.
- Use `apply_patch` for source edits, TDD for every behavior change, and a focused commit after each task.

## File Structure

Create the following focused files:

- `algorithms/matd3_hierarchical_transformer_p2p_value/__init__.py` — public algorithm export only.
- `algorithms/matd3_hierarchical_transformer_p2p_value/plugin.py` — the sole `ALGORITHM_SPEC` registration.
- `algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py` — config validation, typed labels/predictions/losses, physical-unit label extraction, auxiliary head, masked loss, and weight schedule.
- `algorithms/matd3_hierarchical_transformer_p2p_value/actor.py` — predictive actor subclass, shared representation extraction, unchanged policy readout, and P2P prediction entrypoint.
- `algorithms/matd3_hierarchical_transformer_p2p_value/training.py` — parent-equivalent critic/target logic plus combined delayed actor update and sparse auxiliary diagnostics.
- `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py` — child construction, trade-normalizer capture, optimizer/diagnostic cache rebuild, public interface, and strict checkpoint contract.
- `configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml` — isolated copy of the corrected parent experiment settings plus the auxiliary block.
- `scripts/check_p2p_value_gate.py` — saved-result-only primary, safety, and economic gate checker.
- `tests/test_matd3_hierarchical_transformer_p2p_value.py` — labels, actor parity, gradients, training integration, mapping preservation, registry, and checkpoints.
- `tests/test_p2p_value_gate.py` — gate pass/failure and malformed-artifact coverage.

No shared implementation file is modified. If implementation reveals a genuine shared-pipeline defect, stop and document it instead of broadening this stage silently.

## Review Focus

- A future normalization refactor could add offsets: the owning label tests must prove extraction uses the checkpointed physical `trade` scale and rejects a missing/nonpositive scale.
- Replay batches may contain no ESS offers: the owning loss tests must prove both losses and gradients remain finite and exactly zero without manufacturing negative labels.
- Saved outcomes may be corrupt or numerically inconsistent: the owning label tests must exercise NaN, negative energy, accepted-above-offer, and a residual just inside/outside tolerance.
- Self-attention couples every token encoder: the owning actor tests must prove auxiliary gradients reach shared representations but do not directly reach ESS/DR action readout heads or replayed behavior actions.
- Gate artifacts may be incomplete or game acceptance by suppressing volume: the owning gate tests must fail on missing fields/files, low offer energy, low accepted energy, low total P2P energy, and nonfinite metrics.

---

### Task 1: Isolated scaffold, auxiliary configuration, labels, loss, and schedule

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/__init__.py`
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/plugin.py`
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py`
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py`
- Create: `configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml`
- Create: `tests/test_matd3_hierarchical_transformer_p2p_value.py`

**Interfaces:**
- Consumes: observation schema names, normalized next observations, and `normalizers_by_microgrid[agent]["trade"]`.
- Produces: `validate_p2p_auxiliary_config(config: Mapping[str, Any], *, require_enabled: bool) -> dict[str, Any]`, `P2PLabels`, `P2PPrediction`, `P2PLosses`, `P2PValueHead`, `extract_p2p_labels(next_observation, schema, *, trade_scale, offer_epsilon_kwh, consistency_tolerance_kwh) -> P2PLabels`, `compute_p2p_auxiliary_loss(prediction, labels, config) -> P2PLosses`, and `p2p_auxiliary_weight(update_step: int, config) -> float`.

- [ ] **Step 1: Write failing config, label, schedule, and loss tests**

Add imports and tests with the following concrete cases:

```python
from algorithms.matd3_hierarchical_transformer_p2p_value.p2p_auxiliary import (
    P2PPrediction,
    compute_p2p_auxiliary_loss,
    extract_p2p_labels,
    p2p_auxiliary_weight,
    validate_p2p_auxiliary_config,
)

VALID_AUX = {
    "enabled": True,
    "hidden_dim": 64,
    "loss_weight_max": 0.05,
    "ramp_updates": 5000,
    "any_fill_loss_weight": 1.0,
    "fill_ratio_loss_weight": 1.0,
    "offer_epsilon_kwh": 1.0e-6,
    "consistency_tolerance_kwh": 1.0e-5,
    "detach_behavior_action": True,
}

CONFIG_PATH = Path(
    "configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml"
)

@pytest.fixture
def env_and_config():
    env_config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    algo_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return MultiMicrogridEnv(env_config, split="train"), algo_config

def test_label_extraction_restores_kwh_and_masks_zero_offer():
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    next_obs = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.2, 0.3], [1.0, 1.0, 0.0]])
    labels = extract_p2p_labels(
        next_obs, schema, trade_scale=10.0,
        offer_epsilon_kwh=1.0e-6, consistency_tolerance_kwh=1.0e-5,
    )
    assert labels.offered_mask.squeeze(-1).tolist() == [False, True, True]
    assert torch.equal(labels.offer_kwh.squeeze(-1), torch.tensor([0.0, 5.0, 10.0]))
    assert torch.allclose(labels.fill_ratio_target.squeeze(-1), torch.tensor([0.0, 0.4, 1.0]))
    assert labels.any_fill_target.squeeze(-1).tolist() == [0.0, 1.0, 1.0]

@pytest.mark.parametrize("field", ["offer", "accepted", "unmatched"])
def test_label_extraction_rejects_nonfinite_values(field):
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    column = {"offer": 0, "accepted": 1, "unmatched": 2}[field]
    next_obs = torch.tensor([[1.0, 0.5, 0.5]])
    next_obs[0, column] = torch.nan
    with pytest.raises(ValueError, match=field):
        extract_p2p_labels(next_obs, schema, trade_scale=10.0,
                           offer_epsilon_kwh=1e-6,
                           consistency_tolerance_kwh=1e-5)

def test_label_extraction_rejects_negative_and_accepted_above_offer():
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    for row, match in [([-0.1, 0.0, 0.0], "offer"),
                       ([0.5, 0.6, 0.0], "accepted")]:
        with pytest.raises(ValueError, match=match):
            extract_p2p_labels(torch.tensor([row]), schema, trade_scale=10.0,
                               offer_epsilon_kwh=1e-6,
                               consistency_tolerance_kwh=1e-5)

def test_consistency_residual_obeys_tolerance_boundary():
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    scale, tol = 10.0, 1e-5
    boundary = torch.tensor([[1.0, 0.4, (6.0 - tol) / scale]])
    labels = extract_p2p_labels(boundary, schema, trade_scale=scale,
                                offer_epsilon_kwh=1e-6,
                                consistency_tolerance_kwh=tol)
    assert labels.consistency_residual_kwh.item() == pytest.approx(tol, abs=1e-6)
    outside = torch.tensor([[1.0, 0.4, (6.0 - 2 * tol) / scale]])
    with pytest.raises(ValueError, match="consistency"):
        extract_p2p_labels(outside, schema, trade_scale=scale,
                           offer_epsilon_kwh=1e-6,
                           consistency_tolerance_kwh=tol)

def test_label_extraction_rejects_missing_field_and_invalid_trade_scale():
    with pytest.raises(ValueError, match="previous_ess_p2p_accepted.*previous_ess_p2p_unmatched"):
        extract_p2p_labels(torch.zeros(1, 1), ["previous_ess_p2p_offer"],
                           trade_scale=1.0, offer_epsilon_kwh=1e-6,
                           consistency_tolerance_kwh=1e-5)
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="trade_scale"):
            extract_p2p_labels(torch.zeros(1, 3), schema, trade_scale=scale,
                               offer_epsilon_kwh=1e-6,
                               consistency_tolerance_kwh=1e-5)

def test_no_offer_loss_is_differentiable_zero():
    logits = torch.randn(4, 1, requires_grad=True)
    ratios = torch.sigmoid(torch.randn(4, 1, requires_grad=True))
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted",
              "previous_ess_p2p_unmatched"]
    labels = extract_p2p_labels(torch.zeros(4, 3), schema, trade_scale=10.0,
                                offer_epsilon_kwh=1e-6,
                                consistency_tolerance_kwh=1e-5)
    losses = compute_p2p_auxiliary_loss(P2PPrediction(logits, ratios), labels, VALID_AUX)
    losses.total.backward()
    assert losses.total.item() == 0.0
    assert torch.isfinite(logits.grad).all()

def test_auxiliary_weight_linear_ramp():
    assert p2p_auxiliary_weight(0, VALID_AUX) == pytest.approx(0.0)
    assert p2p_auxiliary_weight(2500, VALID_AUX) == pytest.approx(0.025)
    assert p2p_auxiliary_weight(5000, VALID_AUX) == pytest.approx(0.05)
    assert p2p_auxiliary_weight(9000, VALID_AUX) == pytest.approx(0.05)
```

Parameterize config validation to reject `detach_behavior_action=False`, negative weights, nonpositive epsilon/tolerance/hidden size/ramp, booleans supplied as integers, and nonfinite numeric values. Verify `enabled=False` is accepted only with `require_enabled=False`.

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q
```

Expected: collection fails because `algorithms.matd3_hierarchical_transformer_p2p_value` does not exist.

- [ ] **Step 3: Generate the isolated package and config through the repository scaffold**

Run:

```powershell
python scripts/create_algorithm_variant.py `
  --name matd3_hierarchical_transformer_p2p_value `
  --base matd3_hierarchical_transformer_predictive_dr
```

Expected: one new package and `configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml`; no existing algorithm is modified.

Immediately replace the generic scaffold class with a loadable intermediate class that inherits `Matd3HierarchicalTransformerPredictiveDr` without overriding behavior, update `__init__.py` to export `Matd3HierarchicalTransformerP2PValue`, and set the plugin metadata to the exact values specified in Task 3. This keeps the Task 1 commit importable while later tasks add the child actor and checkpoint contract.

- [ ] **Step 4: Implement typed auxiliary primitives and strict validation**

In `p2p_auxiliary.py`, define immutable records with these exact fields:

```python
@dataclass(frozen=True)
class P2PLabels:
    offer_kwh: torch.Tensor
    accepted_kwh: torch.Tensor
    unmatched_kwh: torch.Tensor
    offered_mask: torch.Tensor
    any_fill_target: torch.Tensor
    fill_ratio_target: torch.Tensor
    consistency_residual_kwh: torch.Tensor

@dataclass(frozen=True)
class P2PPrediction:
    any_fill_logit: torch.Tensor
    conditional_fill_ratio: torch.Tensor

@dataclass(frozen=True)
class P2PLosses:
    any_fill: torch.Tensor
    fill_ratio: torch.Tensor
    total: torch.Tensor
    offered_count: torch.Tensor
```

Implement validation with `numbers.Integral`/`numbers.Real`, explicit bool rejection for integer fields, `math.isfinite`, and field-specific error messages. `loss_weight_max`, `any_fill_loss_weight`, and `fill_ratio_loss_weight` may be zero but not negative. `ramp_updates`, `hidden_dim`, epsilon, and tolerance must be strictly positive.

Implement physical label extraction exactly as:

```python
offer_kwh = next_observation[:, offer_idx:offer_idx + 1] * float(trade_scale)
accepted_kwh = next_observation[:, accepted_idx:accepted_idx + 1] * float(trade_scale)
unmatched_kwh = next_observation[:, unmatched_idx:unmatched_idx + 1] * float(trade_scale)
residual = offer_kwh - accepted_kwh - unmatched_kwh
offered = offer_kwh > float(offer_epsilon_kwh)
any_fill = (accepted_kwh > float(offer_epsilon_kwh)).to(next_observation.dtype)
ratio = torch.where(
    offered,
    accepted_kwh / offer_kwh.clamp_min(float(offer_epsilon_kwh)),
    torch.zeros_like(offer_kwh),
).clamp(0.0, 1.0)
```

Reject any nonfinite value, values below `-consistency_tolerance_kwh`, accepted energy greater than offer by more than tolerance, and absolute residual greater than tolerance. Clamp only tiny negative values inside tolerance to zero before producing targets.

Implement the dual head:

```python
class P2PValueHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.any_fill_head = nn.Linear(hidden_dim, 1)
        self.fill_ratio_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, representations, behavior_ess_action):
        behavior = behavior_ess_action.detach()
        features = torch.cat([
            representations.ess,
            representations.supply,
            representations.trade,
            representations.peer,
            behavior,
        ], dim=-1)
        hidden = self.trunk(features)
        return P2PPrediction(self.any_fill_head(hidden), self.fill_ratio_head(hidden))
```

For the masked loss, index only `labels.offered_mask`; if none are true, use `(prediction.any_fill_logit.sum() + prediction.conditional_fill_ratio.sum()) * 0.0`. Otherwise use `binary_cross_entropy_with_logits` and `smooth_l1_loss`, then combine the two configured component weights. Reject nonfinite predictions or losses.

- [ ] **Step 5: Put the exact auxiliary block into the isolated YAML**

Keep every inherited parent field unchanged except `algorithm.name`, then add:

```yaml
  p2p_auxiliary:
    enabled: true
    hidden_dim: 64
    loss_weight_max: 0.05
    ramp_updates: 5000
    any_fill_loss_weight: 1.0
    fill_ratio_loss_weight: 1.0
    offer_epsilon_kwh: 1.0e-6
    consistency_tolerance_kwh: 1.0e-5
    detach_behavior_action: true
```

- [ ] **Step 6: Run the Task 1 tests until they pass**

Run:

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "label or auxiliary_weight or config or no_offer"
```

Expected: all selected tests pass; no warning contains a nonfinite loss.

- [ ] **Step 7: Commit Task 1**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value `
  configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml `
  tests/test_matd3_hierarchical_transformer_p2p_value.py
git commit -m "feat: scaffold P2P value auxiliary learning"
```

---

### Task 2: Child actor, exact policy parity, and gradient boundaries

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/actor.py`
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py`
- Modify: `tests/test_matd3_hierarchical_transformer_p2p_value.py`

**Interfaces:**
- Consumes: `PredictiveDRActor`, `P2PValueHead`, a `[B, obs_dim]` observation, and replayed behavior ESS action `[B, 1]`.
- Produces: `P2PRepresentations`, `P2PValueActor.forward_latent_with_representations(observation: torch.Tensor, collect_diagnostics: bool = False) -> tuple[torch.Tensor, P2PRepresentations]`, unchanged `forward_latent(observation, collect_diagnostics=False) -> torch.Tensor`, and `predict_p2p_value(representations: P2PRepresentations, behavior_ess_action: torch.Tensor) -> P2PPrediction`.

- [ ] **Step 1: Write failing actor shape, parity, and gradient tests**

Add fixtures that construct a parent `PredictiveDRActor` and child `P2PValueActor` with the same schema, environment microgrid config, network sizes, predictive config, and seed. Copy matching parent keys into the child and assert the only missing keys start with `p2p_value_head.`.

```python
def actor_constructor_kwargs(env, config, mg):
    settings = config["algorithm"]
    net = settings["network"]
    normalizers = {
        agent: dict(env.global_state_builder.norm[agent]) for agent in env.agent_ids
    }
    return {
        "schema": env.obs_builder.schema(mg),
        "agent_ids": env.agent_ids,
        "active_agent": mg,
        "mg_config": env.cfg_by_id[mg],
        "predictive_config": settings,
        "normalizers_by_microgrid": normalizers,
        "embed_dim": net["embed_dim"],
        "num_heads": net["num_heads"],
        "num_layers": net["num_layers"],
        "forecast_horizon": net["forecast_horizon"],
        "hidden_dim": net["hidden_dim"],
        "delta_t": env.delta_t,
        "episode_length": env.episode_length,
    }

def make_parent_actor(env_and_config, mg):
    env, config = env_and_config
    return PredictiveDRActor(**actor_constructor_kwargs(env, config, mg))

def make_child_actor(env_and_config, mg, *, enabled):
    env, config = env_and_config
    aux = copy.deepcopy(config["algorithm"]["p2p_auxiliary"])
    aux["enabled"] = enabled
    return P2PValueActor(
        **actor_constructor_kwargs(env, config, mg), p2p_auxiliary_config=aux
    )

def make_child_actor_and_batch(env_and_config, mg, *, batch):
    env, _ = env_and_config
    actor = make_child_actor(env_and_config, mg, enabled=True)
    observations, _ = env.reset(seed=42)
    observation = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
    return actor, observation.repeat(batch, 1)

def module_grad_norm(module):
    squared = [parameter.grad.detach().square().sum()
               for parameter in module.parameters() if parameter.grad is not None]
    return 0.0 if not squared else float(torch.stack(squared).sum().sqrt().item())

@pytest.mark.parametrize("mg", ["mg1", "mg2", "mg3"])
@pytest.mark.parametrize("enabled", [False, True])
def test_child_actor_policy_path_exactly_matches_parent(env_and_config, mg, enabled):
    torch.manual_seed(42)
    parent = make_parent_actor(env_and_config, mg)
    torch.manual_seed(42)
    child = make_child_actor(env_and_config, mg, enabled=enabled)
    incompatible = child.load_state_dict(parent.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(key.startswith("p2p_value_head.") for key in incompatible.missing_keys)
    env, _ = env_and_config
    observations, _ = env.reset(seed=42)
    obs = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
    assert torch.equal(parent.forward_latent(obs), child.forward_latent(obs))
    assert torch.equal(parent(obs), child(obs))

def test_p2p_head_shapes_and_ratio_bounds(env_and_config):
    actor, obs = make_child_actor_and_batch(env_and_config, "mg2", batch=8)
    latent, reps = actor.forward_latent_with_representations(obs)
    pred = actor.predict_p2p_value(reps, torch.linspace(-1, 1, 8).unsqueeze(-1))
    assert latent.shape == (8, 2)
    assert pred.any_fill_logit.shape == (8, 1)
    assert pred.conditional_fill_ratio.shape == (8, 1)
    assert torch.all((pred.conditional_fill_ratio >= 0) & (pred.conditional_fill_ratio <= 1))

def test_auxiliary_only_gradient_boundaries(env_and_config):
    actor, obs = make_child_actor_and_batch(env_and_config, "mg2", batch=8)
    behavior = torch.zeros(8, 1, requires_grad=True)
    _, reps = actor.forward_latent_with_representations(obs)
    pred = actor.predict_p2p_value(reps, behavior)
    (pred.any_fill_logit.mean() + pred.conditional_fill_ratio.mean()).backward()
    assert module_grad_norm(actor.p2p_value_head) > 0
    assert module_grad_norm(actor.trade_encoder) > 0
    assert module_grad_norm(actor.transformer) > 0
    assert module_grad_norm(actor.ess_head) == 0
    assert module_grad_norm(actor.ess_linear_skip) == 0
    assert module_grad_norm(actor.dr_head) == 0
    assert module_grad_norm(actor.dr_linear_skip) == 0
    assert behavior.grad is None
```

Also assert malformed observation rank fails with the same parent-style message and a child built from a schema missing any required P2P field names the complete missing-field list.

- [ ] **Step 2: Run actor tests and verify failure**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "child_actor or p2p_head or gradient_boundaries"
```

Expected: import or attribute failures for `P2PValueActor` and `P2PRepresentations`.

- [ ] **Step 3: Implement representation extraction without changing the action computation**

Define in `p2p_auxiliary.py`:

```python
@dataclass(frozen=True)
class P2PRepresentations:
    ess: torch.Tensor
    supply: torch.Tensor
    trade: torch.Tensor
    peer: torch.Tensor
```

In `actor.py`, subclass `PredictiveDRActor`. After `super().__init__`, validate the three previous P2P observation names, retain the canonicalized auxiliary config, and construct `P2PValueHead(self.embed_dim, hidden_dim)`.

Implement one shared method that copies the parent's policy operations exactly and returns both outputs:

```python
def forward_latent_with_representations(self, observation, collect_diagnostics=False):
    if observation.ndim != 2:
        raise ValueError(
            f"Expected observation shape [batch_size, {len(self.schema)}], "
            f"got {observation.shape}"
        )
    token_tensor, padding_mask = self._build_tokens(observation)
    attended = self.transformer(token_tensor, src_key_padding_mask=padding_mask)
    ess_rep, dr_rep = attended[:, 0, :], attended[:, 1, :]
    supply_rep, trade_rep = attended[:, 2, :], attended[:, 3, :]
    peer_rep = attended[:, 4:4 + self.num_peers, :].mean(dim=1)

    ess_feats = observation.index_select(-1, self.ess_idx_tensor)
    dr_feats = observation.index_select(-1, self.dr_idx_tensor)
    a_ess = self.ess_head(torch.cat([
        ess_rep, supply_rep, trade_rep, self.ess_linear_skip(ess_feats)
    ], dim=-1))
    a_dr = self.dr_head(torch.cat([
        dr_rep, supply_rep, trade_rep, self.dr_linear_skip(dr_feats)
    ], dim=-1))
    latent = torch.cat([a_ess, a_dr], dim=-1)
    # Preserve the parent's diagnostic keys and values when requested.
    return latent, P2PRepresentations(ess_rep, supply_rep, trade_rep, peer_rep)
```

`forward_latent` delegates to this method and returns only `latent`. `predict_p2p_value` accepts already-computed representations so training never performs a second Transformer pass. `forward` remains inherited from `PredictiveDRActor`, therefore predictive and feasible mapping still execute in the same place.

Extend `branch_modules()` by copying the parent mapping and adding `"p2p_auxiliary": [self.p2p_value_head]`; do not put the auxiliary head into either action-head branch.

- [ ] **Step 4: Run parity and gradient tests**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "child_actor or p2p_head or gradient_boundaries"
```

Expected: exact latent/action parity for all three agents, bounded ratio outputs, shared gradients present, action-head and behavior-action gradients absent.

- [ ] **Step 5: Commit Task 2**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value/actor.py `
  algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py `
  tests/test_matd3_hierarchical_transformer_p2p_value.py
git commit -m "feat: add action-conditioned P2P value head"
```

---

### Task 3: Algorithm construction, registry, inherited mapping, and strict checkpoints

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/__init__.py`
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/plugin.py`
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_p2p_value.py`

**Interfaces:**
- Consumes: corrected parent algorithm `Matd3HierarchicalTransformerPredictiveDr`, child actors, environment schemas/configuration, source normalizers, and the isolated YAML.
- Produces: `Matd3HierarchicalTransformerP2PValue` implementing `from_env`, `select_actions`, `update`, `save`, `load`, `replay.add`, and `len(replay)`.

- [ ] **Step 1: Write failing construction, registry, mapping, and checkpoint tests**

Add tests that:

1. resolve `get_algorithm_spec("matd3_hierarchical_transformer_p2p_value")` and assert the slug, class, and exact YAML path;
2. build through `from_env`, assert every actor is `P2PValueActor`, and assert each stored normalizer contains positive `trade`;
3. compare deterministic child actions to a parent initialized with copied inherited weights;
4. assert MG1 mapping remains identity for DR, MG2 stays inside predictive HVAC bounds, MG3 stays inside feasible EV bounds, and unavailable MG3 returns exactly raw `-1.0`;
5. save/load a round trip and assert actor/target/critic tensors and `update_steps` are equal;
6. mutate each strict field and assert the error names that field.

Add a deterministic 24-hour closed-loop preservation test using equal inherited weights, identical environment seeds, and `noise_std=0.0`. Assert parent and child hourly actions are exactly equal before training, then build the saved rollout constraint summary and assert MG2/MG3 projection rates are both zero, MG3 departure success is `1.0`, and MG1 terminal-or-production success is at least `0.995`.

Use the following mutation table:

```python
[
    ("algorithm_slug", "matd3_hierarchical_transformer_predictive_dr", "algorithm_slug"),
    ("action_semantics", "wrong", "action_semantics"),
    ("parent_predictive_contract_version", "9.9", "parent_predictive_contract_version"),
    ("p2p_auxiliary_contract_version", "9.9", "p2p_auxiliary_contract_version"),
    ("predictive_hvac_config", {"enabled": False}, "predictive_hvac_config"),
    ("feasible_dr_config", {"enabled": False}, "feasible_dr_config"),
    ("p2p_auxiliary_config", {"enabled": False}, "p2p_auxiliary_config"),
    ("observation_schema_by_agent", {"mg1": ["wrong"]}, "observation_schema_by_agent"),
    ("normalizers_by_microgrid", {"mg1": {"trade": 1.0}}, "normalizers_by_microgrid"),
]
```

Also load a real parent checkpoint into the child and assert rejection occurs before any state dictionary is applied.

- [ ] **Step 2: Run the construction/checkpoint tests and verify scaffold failure**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "registry or algorithm or mapping or checkpoint"
```

Expected: failures because the scaffold algorithm still inherits canonical MATD3 and the plugin metadata is generic.

- [ ] **Step 3: Implement child algorithm construction**

Define constants:

```python
ALGORITHM_SLUG = "matd3_hierarchical_transformer_p2p_value"
PARENT_PREDICTIVE_CONTRACT_VERSION = "1.1"
P2P_AUXILIARY_CONTRACT_VERSION = "1.0"
```

The class must:

1. deep-copy and validate `settings["p2p_auxiliary"]` with `require_enabled=True` before training;
2. call the corrected parent constructor and assert `self.predictive_contract_version == "1.1"`;
3. copy `load`, `pv`, `wind`, `power`, `price`, and `trade` from `env.global_state_builder.norm`, validating every value is finite and positive;
4. recompute the child normalization digest from this full mapping;
5. call `torch.manual_seed(seed)`, construct a `ModuleDict` of `P2PValueActor`, deep-copy it for targets, and disable target gradients;
6. rebuild Adam actor optimizers and all `_actor_parameters`, `_actor_branch_parameters`, and `_actor_layer_parameters` caches against the new actors;
7. retain the parent's critics, projector, replay, diagnostic schedule, RNG, and public action selector;
8. implement `update` by delegating to Task 4's `update_p2p_value`.

Use `action_semantics = "single_dispatch_dr_predictive_p2p_value"`. `select_actions` delegates to the existing `select_predictive_actions`, preserving latent-space noise and physical mapping.

- [ ] **Step 4: Implement exact plugin and public export**

`plugin.py` must contain only one `ALGORITHM_SPEC`:

```python
ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_p2p_value",
    display_name="MATD3 Hierarchical Transformer P2P Value",
    algorithm_class=Matd3HierarchicalTransformerP2PValue,
    default_config_path=(PROJECT_ROOT / "configs" / "algorithms" /
                         "matd3_hierarchical_transformer_p2p_value.yaml"),
    description=("Predictive feasible-DR hierarchical Transformer with an "
                 "action-conditioned P2P fill-value auxiliary objective."),
)
```

`__init__.py` exports only `Matd3HierarchicalTransformerP2PValue`.

- [ ] **Step 5: Implement strict save/load before state mutation**

Save the following contract fields plus model tensors and `update_steps`:

```python
{
    "algorithm_slug": ALGORITHM_SLUG,
    "action_semantics": self.action_semantics,
    "parent_predictive_contract_version": PARENT_PREDICTIVE_CONTRACT_VERSION,
    "predictive_hvac_config": self.canonical_predictive_config(
        self.predictive_hvac_config
    ),
    "feasible_dr_config": self.canonical_predictive_config(
        self.feasible_dr_config
    ),
    "p2p_auxiliary_contract_version": P2P_AUXILIARY_CONTRACT_VERSION,
    "p2p_auxiliary_config": self.canonical_predictive_config(
        self.p2p_auxiliary_config
    ),
    "observation_schema_by_agent": {
        mg: list(self.observation_schemas[mg]) for mg in self.agent_ids
    },
    "normalizers_by_microgrid": self.normalizers_by_microgrid,
    "normalization_signature": self.normalization_signature,
    "actors": {mg: self.actors[mg].state_dict() for mg in self.agent_ids},
    "target_actors": {
        mg: self.target_actors[mg].state_dict() for mg in self.agent_ids
    },
    "critics": {mg: self.critics[mg].state_dict() for mg in self.agent_ids},
    "target_critics": {
        mg: self.target_critics[mg].state_dict() for mg in self.agent_ids
    },
    "update_steps": self.update_steps,
}
```

In `load`, validate every scalar/config/schema/normalizer contract field first. Only after all checks pass may any actor or critic call `load_state_dict`. Missing fields are mismatches rather than defaults.

- [ ] **Step 6: Run the Task 3 tests**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "registry or algorithm or mapping or checkpoint"
```

Expected: all selected tests pass, including parent-checkpoint rejection and unavailable-EV behavior.

- [ ] **Step 7: Commit Task 3**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value `
  tests/test_matd3_hierarchical_transformer_p2p_value.py
git commit -m "feat: isolate P2P value algorithm checkpoints"
```

---

### Task 4: Combined actor update and sparse P2P diagnostics

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_p2p_value/training.py`
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_p2p_value.py`

**Interfaces:**
- Consumes: `FastReplayBuffer` batches, child actors, inherited critics/projector/target actors, `extract_p2p_labels`, and `compute_p2p_auxiliary_loss`.
- Produces: `update_p2p_value(self, batch_size) -> dict[str, Any]` with parent diagnostics plus aggregate and per-agent `p2p_aux_*` metrics.

- [ ] **Step 1: Write failing loss-integration, cadence, finiteness, and diagnostics tests**

Create deterministic batches with offered rows by taking real reset observations, writing the three next-observation P2P columns to normalized `(offer, accepted, unmatched) = (10, 4, 6) kWh`, and adding enough transitions for a small batch.

Add these exact test helpers so every integration test owns its batch construction:

```python
def clone_parameters(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}

def parameters_changed(before, module):
    return any(not torch.equal(before[name], value.detach())
               for name, value in module.named_parameters())

def add_replay_rows(algo, env, *, count, outcome_kwh):
    observations, reset_info = env.reset(seed=123)
    state = reset_info["global_state"]
    for row_index in range(count):
        actions = {mg: np.zeros(env.action_dim, dtype=np.float32)
                   for mg in env.agent_ids}
        next_observations, rewards, done, info = env.step(actions)
        patched = {mg: value.copy() for mg, value in next_observations.items()}
        for mg in env.agent_ids:
            scale = algo.normalizers_by_microgrid[mg]["trade"]
            schema = algo.observation_schemas[mg]
            for name, physical in zip(
                ("previous_ess_p2p_offer", "previous_ess_p2p_accepted",
                 "previous_ess_p2p_unmatched"), outcome_kwh
            ):
                patched[mg][schema.index(name)] = physical / scale
        algo.replay.add(observations, state, actions,
                        float(rewards[env.agent_ids[0]]), patched,
                        info["global_state_after"], done)
        if done and row_index + 1 < count:
            observations, reset_info = env.reset(seed=124 + row_index)
            state = reset_info["global_state"]
        else:
            observations, state = next_observations, info["global_state_after"]
```

Add tests with concrete fixtures and assertions:

```python
def test_update_changes_auxiliary_and_shared_parameters_on_actor_step(env_and_config):
    env, config = env_and_config
    config = copy.deepcopy(config)
    config["algorithm"]["policy_delay"] = 1
    config.setdefault("logging", {})["diagnostic_interval"] = 1
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    add_replay_rows(algo, env, count=4, outcome_kwh=(10.0, 4.0, 6.0))
    algo.policy_delay = 1
    before_head = clone_parameters(algo.actors["mg2"].p2p_value_head)
    before_trade = clone_parameters(algo.actors["mg2"].trade_encoder)
    metrics = algo.update(batch_size=4)
    assert parameters_changed(before_head, algo.actors["mg2"].p2p_value_head)
    assert parameters_changed(before_trade, algo.actors["mg2"].trade_encoder)
    assert algo.update_steps == 1
    assert metrics["p2p_aux_offered_sample_count"] == pytest.approx(4.0)

def test_no_offer_training_batch_is_finite(env_and_config):
    env, config = env_and_config
    config = copy.deepcopy(config)
    config["algorithm"]["policy_delay"] = 1
    config.setdefault("logging", {})["diagnostic_interval"] = 1
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    add_replay_rows(algo, env, count=4, outcome_kwh=(0.0, 0.0, 0.0))
    metrics = algo.update(batch_size=4)
    assert metrics["p2p_aux_total_loss"] == pytest.approx(0.0)
    assert metrics["p2p_aux_offered_sample_count"] == pytest.approx(0.0)
    assert all(torch.isfinite(parameter).all() for parameter in algo.actors.parameters())
```

For zero-weight parent equivalence, build parent and child with the same seed, set the child `loss_weight_max` to `0.0`, copy every matching actor/target/critic tensor, give both replay buffers identical transitions, capture `torch.random.get_rng_state()`, restore it before each update, and compare every inherited named parameter exactly after one actor-due update. Ignore only `p2p_value_head.*`, which must remain unchanged.

For diagnostic non-interference, construct two child algorithms differing only in `logging.diagnostic_interval` (`0` versus `1`), synchronize model/replay/RNG state exactly as above, run one update, and use:

```python
for left, right in zip(algo_without_diagnostics.actors.parameters(),
                       algo_with_diagnostics.actors.parameters()):
    assert torch.equal(left, right)
for left, right in zip(algo_without_diagnostics.critics.parameters(),
                       algo_with_diagnostics.critics.parameters()):
    assert torch.equal(left, right)
```

For critic equivalence, set `policy_delay=2`, synchronize a parent and zero-weight child, perform only update step `1` so no actor is due, and compare all critic/target-critic tensors exactly. At diagnostic cadence, assert every metric named in Step 4 exists, is finite, and has a per-agent mapping. Inject a nonfinite prediction with this exact replacement and assert `FloatingPointError` occurs before any pre-captured actor parameter changes:

```python
def nonfinite_prediction(representations, behavior_ess_action):
    batch = behavior_ess_action.shape[0]
    device = behavior_ess_action.device
    return P2PPrediction(
        torch.full((batch, 1), float("nan"), device=device),
        torch.zeros((batch, 1), device=device),
    )

algo.actors["mg2"].p2p_value_head.forward = nonfinite_prediction
```

- [ ] **Step 2: Run the training tests and verify failure**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "update or diagnostics or target_q or no_offer_training"
```

Expected: failures because `update_p2p_value` is not implemented.

- [ ] **Step 3: Copy the corrected parent update locally, then make only the actor-side delta**

Start from `update_predictive_dr` in the parent package. Keep replay sampling, projection, target latent noise, target mapping, target-Q, twin critic losses, critic optimizer cadence, gradient clipping, target soft updates, and existing actor diagnostics byte-for-byte equivalent where practical.

For each actor update, compute policy actions and representations once:

```python
latent, reps = actor.forward_latent_with_representations(
    batch["observations"][mg], collect_diagnostics=diagnostic_due
)
current_actions[mg] = actor.map_latent_actions(
    batch["observations"][mg], latent, collect_diagnostics=diagnostic_due
)
representations[mg] = reps
```

For the active actor, extract labels using its own schema and physical trade scale, then combine losses:

```python
policy_loss = -critic.q1(state, projected.flatten(1)).mean()
prediction = actor.predict_p2p_value(
    representations[active], batch["actions"][active][:, 0:1]
)
labels = extract_p2p_labels(
    batch["next_observations"][active],
    self.observation_schemas[active],
    trade_scale=self.normalizers_by_microgrid[active]["trade"],
    offer_epsilon_kwh=self.p2p_auxiliary_config["offer_epsilon_kwh"],
    consistency_tolerance_kwh=self.p2p_auxiliary_config["consistency_tolerance_kwh"],
)
aux = compute_p2p_auxiliary_loss(prediction, labels, self.p2p_auxiliary_config)
weight = p2p_auxiliary_weight(next_update_step, self.p2p_auxiliary_config)
total_loss = policy_loss + weight * aux.total
```

Zero gradients once, backpropagate once through `total_loss`, check every populated gradient and the preclip norm for finiteness, clip once, step once, and soft-update actor/critic targets once. Do not perform an auxiliary optimizer step.

- [ ] **Step 4: Add sparse diagnostics without affecting computation**

At `diagnostic_due`, derive detached values from already-computed tensors:

```text
p2p_auxiliary_weight
p2p_aux_any_fill_loss
p2p_aux_fill_ratio_loss
p2p_aux_total_loss
p2p_aux_offered_sample_rate
p2p_aux_offered_sample_count
p2p_aux_positive_fill_rate
p2p_aux_fill_ratio_mae
p2p_aux_predicted_fill_mean
p2p_aux_actual_fill_mean
p2p_aux_offer_consistency_error_mean
p2p_aux_shared_gradient_norm
p2p_aux_head_gradient_norm
```

Store per-agent values under `p2p_auxiliary_diagnostics` and top-level means under the exact names above. For zero offered rows, all conditional means and MAE are `0.0`. Compute gradient norms after backward and before clipping; do not run another forward or backward pass for diagnostics.

- [ ] **Step 5: Run focused training tests**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q -k "update or diagnostics or target_q or no_offer_training"
```

Expected: combined update, zero-weight equivalence, no-offer stability, target/critic equivalence, and diagnostic non-interference all pass.

- [ ] **Step 6: Run the entire new algorithm test file**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -q
```

Expected: all label, actor, mapping, training, registry, and checkpoint tests pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value/training.py `
  algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py `
  tests/test_matd3_hierarchical_transformer_p2p_value.py
git commit -m "feat: train P2P value head with masked outcomes"
```

---

### Task 5: Saved-result P2P acceptance and safety gate

**Files:**
- Create: `scripts/check_p2p_value_gate.py`
- Create: `tests/test_p2p_value_gate.py`

**Interfaces:**
- Consumes: saved candidate, frozen Transformer reference, and predictive exploratory run artifacts only.
- Produces: `evaluate_p2p_value_gate(candidate_dir, reference_dir, predictive_dir, atol=1e-9) -> dict[str, Any]` and a CLI returning `0` for pass, `1` for gate failure, and `2` for malformed input.

- [ ] **Step 1: Write failing gate tests with self-contained saved artifacts**

Create a helper that writes minimal `run_metadata.json`, `training_summary.json`, `test_summary.json`, `paper_metrics.json`, `dr_shift_metrics.csv`, and hourly trajectory CSVs. The candidate slug is exactly `matd3_hierarchical_transformer_p2p_value`; `checkpoint_used` ends in `checkpoints/best_model.pt`.

Use this helper interface and nested artifact shape so the tests exercise the production reader rather than mocks:

```python
def write_saved_run(
    root, *, slug, acceptance, offer, accepted, total_p2p,
    strict_feasible=1.0, unmet=0.0, comfort_pct=99.6,
    hvac_joint_pct=50.0, hvac_late_power_p95=5.0,
    mill_terminal_pct=100.0, mill_production_pct=100.0,
    ev_pct=100.0, mg2_projection_pct=0.0, mg3_projection_pct=0.0,
    cost=870.0, renewable=0.97, peak_hourly=40.0,
    checkpoint="checkpoints/best_model.pt",
):
    root.mkdir(parents=True)
    (root / "run_metadata.json").write_text(
        json.dumps({"algorithm_slug": slug, "run_id": root.name}), encoding="utf-8"
    )
    (root / "training_summary.json").write_text(json.dumps({
        "checkpoint_selection": {
            "selection_mode": "constrained_profit",
            "constraint_eligible": True,
            "best_checkpoint": checkpoint,
        }
    }), encoding="utf-8")
    per_agent_offer = {"mg1": offer / 3, "mg2": offer / 3, "mg3": offer / 3}
    per_agent_accepted = {
        "mg1": accepted / 3, "mg2": accepted / 3, "mg3": accepted / 3
    }
    (root / "test_summary.json").write_text(json.dumps({
        "checkpoint_used": checkpoint,
        "strict_feasible_day_rate": strict_feasible,
        "unmet_energy_mean_kwh": unmet,
        "renewable_utilization_rate": renewable,
        "p2p_total_energy_kwh_mean": total_p2p,
        "algorithm": {"p2p": {
            "total_energy_mean_kwh": total_p2p,
            "ess_offer": {
                "offer_mean_kwh": per_agent_offer,
                "accepted_mean_kwh": per_agent_accepted,
                "energy_acceptance_rate": acceptance,
            },
        }},
    }), encoding="utf-8")
    metrics = {
        "hvac_terminal_joint_success_rate": hvac_joint_pct,
        "hvac_late_power_p95_kw": hvac_late_power_p95,
        "hvac_occupied_comfort_rate": comfort_pct,
        "mill_terminal_success_rate": mill_terminal_pct,
        "mill_production_completion_rate": mill_production_pct,
        "ev_departure_target_success_rate": ev_pct,
        "mg2_dr_projection_rate": mg2_projection_pct,
        "mg3_dr_projection_rate": mg3_projection_pct,
        "total_operating_cost_day": cost,
    }
    (root / "paper_metrics.json").write_text(json.dumps({
        "metrics": [{"metric_id": key, "value": value}
                    for key, value in metrics.items()]
    }), encoding="utf-8")
    (root / "dr_shift_metrics.csv").write_text(
        "date,peak_load_reduction_pct\n2018-04-27,5.0\n", encoding="utf-8"
    )
    hourly = root / "test_hourly_trajectories"
    hourly.mkdir()
    rows = ["date,hour,grid_import_kwh\n"] + [
        f"2018-04-27,{hour},{peak_hourly if 16 <= hour <= 21 else 1.0}\n"
        for hour in range(24)
    ]
    (hourly / "2018-04-27.csv").write_text("".join(rows), encoding="utf-8")
    return root
```

Test a passing candidate with:

```text
acceptance rate = 0.37
offer energy = 340 kWh/day
accepted energy = 145 kWh/day
total P2P = 1060 kWh/day
strict feasible = 1.0
unmet energy = 0.0
MG2/MG3 projection = 0.0
EV departure = 1.0
occupied comfort = 0.996
MG1 mill terminal or production = 1.0
```

Use reference values matching the frozen baseline scale and predictive acceptance `0.3127`. Assert separate failures for acceptance below `0.3649`, improvement below `0.05`, offer volume below `80%` of reference, accepted volume below `95%`, total P2P below `95%`, each safety gate, cost above `102%`, peak import above `105%`, and renewable utilization more than `0.5` percentage points lower.

Add malformed-artifact tests for missing files, missing nested P2P keys, nonfinite values, wrong slug, and a non-best checkpoint.

The passing test calls `write_saved_run` three times: candidate values from the block above, frozen-reference values `(acceptance=0.364865, offer=412.65, accepted=150.56, total_p2p=1105.03)`, and predictive values `(acceptance=0.312712, offer=427.62, accepted=133.72, total_p2p=1120.66)`. It then asserts `overall_status == "PASS"` and every condition is true.

Use a parameterized failure table with exact overrides and condition keys:

```python
@pytest.mark.parametrize(("override", "condition"), [
    ({"acceptance": 0.36}, "p2p_acceptance_ge_36_49pct"),
    ({"acceptance": 0.35}, "p2p_acceptance_improvement_ge_5pp"),
    ({"offer": 320.0}, "p2p_offer_volume_ge_80pct_reference"),
    ({"accepted": 140.0}, "p2p_accepted_volume_ge_95pct_reference"),
    ({"total_p2p": 1000.0}, "total_p2p_ge_95pct_reference"),
    ({"strict_feasible": 0.99}, "strict_feasible_rate_100pct"),
    ({"unmet": 0.01}, "unmet_energy_eq_0_kwh"),
    ({"comfort_pct": 99.4}, "hvac_comfort_ge_99_5pct"),
    ({"hvac_joint_pct": 44.9}, "hvac_joint_no_worse_than_predictive_5pp"),
    ({"hvac_late_power_p95": 5.3}, "hvac_late_power_no_worse_than_predictive_5pct"),
    ({"mill_terminal_pct": 99.0, "mill_production_pct": 99.0},
     "mill_terminal_or_production_ge_99_5pct"),
    ({"ev_pct": 99.0}, "ev_departure_success_100pct"),
    ({"mg2_projection_pct": 0.1}, "mg2_projection_rate_0"),
    ({"mg3_projection_pct": 0.1}, "mg3_projection_rate_0"),
    ({"cost": 950.0}, "economic_cost_le_102pct_reference"),
    ({"peak_hourly": 50.0}, "economic_peak_import_le_105pct_reference"),
    ({"renewable": 0.96}, "economic_res_utilization_within_0_5pct"),
])
def test_each_gate_fails_independently(saved_gate_fixture, override, condition):
    candidate, reference, predictive = saved_gate_fixture(**override)
    report = evaluate_p2p_value_gate(candidate, reference, predictive)
    assert report["overall_status"] == "FAIL"
    assert report["conditions"][condition]["passed"] is False
```

Malformed tests delete one required file, remove `algorithm.p2p.ess_offer`, write `NaN`, change the candidate slug, and replace the checkpoint path; each must raise a field-specific `FileNotFoundError` or `ValueError` from `evaluate_p2p_value_gate`.

- [ ] **Step 2: Run gate tests and verify import failure**

```powershell
pytest tests/test_p2p_value_gate.py -q
```

Expected: collection fails because `scripts.check_p2p_value_gate` does not exist.

- [ ] **Step 3: Implement a saved-result-only reader and gate**

Reuse pure helpers from `scripts.check_predictive_hvac_dr_gate` only when their semantics match. Never import the environment, algorithm, evaluator, or MILP code.

Set exact defaults:

```python
DEFAULT_REFERENCE_RUN = Path(
    "outputs/working_runs/matd3_hierarchical_transformer/runs/"
    "MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42"
)
DEFAULT_PREDICTIVE_RUN = Path(
    "outputs/working_runs/matd3_hierarchical_transformer_predictive_dr/runs/"
    "MATD3HIERARCHICALTRANSFORMERPREDICTIVEDR_2026-09-21_13-01-58_seed42"
)
```

Read candidate offer and accepted energy by summing `algorithm.p2p.ess_offer.offer_mean_kwh` and `accepted_mean_kwh` across all agents. Read total P2P from `algorithm.p2p.total_energy_mean_kwh`, and require it to agree with top-level `p2p_total_energy_kwh_mean` within tolerance when both exist.

Evaluate every condition independently and return the measured value, comparison baseline, threshold, and boolean. Overall status is `FAIL` if any primary, safety, checkpoint, or economic condition fails.

- [ ] **Step 4: Implement the CLI**

Support:

```powershell
python scripts/check_p2p_value_gate.py `
  --run <candidate_run> `
  --reference <reference_run> `
  --predictive <predictive_run> `
  --json
```

The CLI catches malformed artifacts, writes a concise error to stderr, and returns `2`. JSON output includes source hashes so the gate report identifies immutable evidence.

- [ ] **Step 5: Run gate tests**

```powershell
pytest tests/test_p2p_value_gate.py -q
```

Expected: all primary, safety, economic, anti-gaming, and malformed-input cases pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add scripts/check_p2p_value_gate.py tests/test_p2p_value_gate.py
git commit -m "feat: add saved-result P2P value gate"
```

---

### Task 6: Full regression, shared-pipeline smoke test, and evidence audit

**Files:**
- Modify only if verification exposes a defect: files created in Tasks 1–5 and their tests.
- Verify: `outputs/working_runs/matd3_hierarchical_transformer_p2p_value/runs/<smoke_run>/`

**Interfaces:**
- Consumes: the registered child algorithm and shared `scripts/run_experiment.py` workflow.
- Produces: a complete isolated smoke run and reproducible verification evidence; no release promotion and no 5000-episode training.

- [ ] **Step 1: Run focused parent and child regression tests**

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py `
  tests/test_p2p_value_gate.py `
  tests/test_predictive_hvac.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the full repository suite**

```powershell
pytest -q
```

Expected: all tests pass with no new failure; record the exact count, warnings, and elapsed time.

- [ ] **Step 3: Prove existing algorithm source packages are unchanged**

Run:

```powershell
git diff fbf79e2 -- algorithms/matd3 `
  algorithms/matd3_hierarchical_transformer `
  algorithms/matd3_hierarchical_transformer_feasible_dr `
  algorithms/matd3_hierarchical_transformer_predictive_dr
```

Expected: no output. Also inspect `git status --short` and ensure only intentional child/gate/test changes remain before the verification commit.

- [ ] **Step 4: Run a three-episode CPU shared-pipeline smoke test**

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_p2p_value `
  --episodes 3 `
  --test-episodes 3 `
  --validation-interval 1 `
  --validation-days 2 `
  --device cpu `
  --seed 42 `
  --no-dashboard
```

Expected: training, validation, best-checkpoint selection, deterministic formal test, daily coalition MILP, Equal/Shapley settlement, summaries, and saved-result plotting complete under the new slug.

- [ ] **Step 5: Audit the smoke artifacts without rerunning evaluation**

Resolve the new `latest_run.json`, then verify:

- `run_metadata.json` and `config_snapshot.yaml` name `matd3_hierarchical_transformer_p2p_value`;
- `checkpoints/best_model.pt` and `checkpoints/final_model.pt` exist;
- `test_summary.json` reports `best_model.pt` as `checkpoint_used`;
- training and validation histories, summaries, test daily metrics, hourly trajectories, settlement artifacts, and coalition MILP outputs exist;
- `figures/` contains exactly 15 PNG and 15 PDF files unless the shared pipeline's current documented requirement has changed;
- the checkpoint contains both contract versions, the full P2P config, schemas, and `trade` normalizers;
- actor diagnostics include finite `p2p_aux_*` values once replay warmup and actor update conditions permit them;
- no output was written into another algorithm's result directory.

Because a three-episode run is shorter than the default warmup of `1000` steps, absence of learned auxiliary diagnostics is acceptable only if the saved config shows that no optimizer update occurred. Unit/integration tests remain the proof of auxiliary training behavior.

- [ ] **Step 6: Run the gate CLI in malformed/smoke mode, not as a scientific claim**

Run the checker against the smoke run and expect either gate `FAIL` (exit `1`) because three episodes cannot establish performance, or a clear artifact error if the smoke result intentionally omits a full-study artifact. The command must not crash or mutate saved files. Do not report a smoke gate pass as evidence of improvement.

- [ ] **Step 7: Fix only child-local defects and repeat affected verification**

For any failure, first reproduce it with the narrowest test, patch only the files created in Tasks 1–5, rerun that test, then rerun the full suite and smoke step affected by the change. Do not weaken assertions or acceptance thresholds.

- [ ] **Step 8: Commit verification fixes/evidence-ready code**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value `
  configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml `
  scripts/check_p2p_value_gate.py `
  tests/test_matd3_hierarchical_transformer_p2p_value.py `
  tests/test_p2p_value_gate.py
git commit -m "test: verify P2P value algorithm pipeline"
```

If there are no code changes after the Task 5 commit, do not create an empty commit; record verification results in the final handoff instead.

---

## Completion Handoff

Report:

- commit hashes for every implemented task;
- exact focused and full-suite test results;
- smoke run absolute path and artifact counts;
- confirmation that the formal smoke test used `best_model.pt`;
- confirmation that canonical MATD3 and all parent experimental packages have zero diff from `fbf79e2`;
- any warnings or limitations, especially that the smoke run does not prove P2P improvement;
- the exact clean command for the later 5000-episode seed-42 run, but do not start it unless the user asks.
