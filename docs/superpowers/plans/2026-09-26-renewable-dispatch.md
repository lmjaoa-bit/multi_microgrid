# Renewable dispatch environment implementation plan

> **Superseded status (2026-09-27):** This 2026-09-26 plan is retained as historical design evidence. The authoritative implementation plan is docs/superpowers/plans/2026-09-27-renewable-first-dispatch-and-metrics.md with its linked design spec. In particular, the current v3 dispatch is renewable-load-first rather than the earlier local-first ordering described below.


> **Handoff update (2026-09-26):** The user stopped implementation and requested a plan for another developer. The authoritative consolidated handoff is [实施交接_新能源协同调度.md](../../实施交接_新能源协同调度.md). Existing source edits are incomplete and unverified. The execution notes below describe the earlier implementation phase, not authorization to resume it in this documentation turn.

> **For agentic workers:** Execute the approved user design with test-driven changes. Follow executing-plans; use parallel domain work only with disjoint file ownership.

**Goal:** Implement the user's final shared environment: natural days, independent initial states, three EV cohorts, common existing tariff, unlimited grid purchases, 100 kW exports, unconstrained source-aware P2P, two-action local-first ESS/DR.

**Architecture:** Keep the shared pipeline and canonical MATD3 unchanged. Device state, energy clearing, and optimization each implement the same documented physics. Use explicit null for unbounded configurable capacities; finite physical derived bounds in MILP avoid infinite big-M coefficients.

**Tech Stack:** Python, NumPy, pandas, PyTorch, HiGHS, pytest.

**Spec:** Approved conversation plus `../specs/2026-09-26-mg3-ev-cohorts-design.md`. Later conversation supersedes grid/ESS assumptions in that document.

## Global constraints and final decisions

- Raw actor action stays `[ESS, DR]`; ESS remains local-first; no canonical `algorithms/matd3/` changes.
- All microgrids inherit existing data `buy_price_per_kwh`; sell price is 0.5 times buy. No new TOU curve.
- Grid import and P2P pair limits are null (unlimited), grid export is 100 kW per microgrid, loss is zero.
- Renewable supply can serve local/peer loads and ESS; ESS discharge serves local/peer loads only. No direct grid resale, no ESS-to-ESS, no ESS-to-grid, no routing imports to export.
- Match load before charging; renewable before ESS within load clearing. Unmatched ESS offers do not discharge. Grid serves all remaining legal demand.
- ESS reset SOC equals minimum 0.1; no terminal restoration. MG1 retains initial silo 10.53 t and continuous downstream service with no terminal restoration. HVAC retains existing weather-conditioned initial temperatures and occupied/unoccupied comfort, no terminal restoration.
- Three EV groups use the detailed spec: 6/4/3 equivalent vehicles, 60 kWh, 7.2 kW each, eta .93, per-car need Uniform(14,22), arrivals/departures as specified. Deterministic date + configured scenario seed generation shared with MILP; no dependence on policy RNG. Observations reveal future group only on arrival. One EV total-power action maps to feasible [L,U], with LLF allocation after minimum reservations. Departure shortfall charged once at 1.5/kWh independently of midnight penalties.
- Metrics use curtailment, non-curtailment, exports, imports, actual cost, service completion and end inventories; do not claim a final-use renewable share from battery throughput.
- MILP uses the same sessions and source/use constraints; label perfect foresight and free cohort allocation as oracle advantages.
- Existing finite-limit and single-fleet configurations remain readable for regression/reproducibility where practical.

## Tasks

### 1. EV device and observation contract
- [ ] Write failing tests for date-stable cohorts, hidden arrivals, last charging interval, mandatory floor, actual delivered energy and once-only departure shortfall.
- [ ] Implement cohort model behind EVChargingDR with legacy fallback; expose `sessions` (list of dictionaries with group_id, arrival_hour, departure_hour, demand_kwh, capacity_kwh, max_charge_kw, initial_soc, target_soc), cohort state fields, and departure events via commit.
- [ ] Use `scenario_seed` default 42 in device config, stable calendar date; no true future observations. Add 28 cohort features (8/group + remaining/L/U/station) with consistent local and global schema.
- [ ] Update ordinary T input adaptation and EV rule reference, retaining canonical algorithms.

### 2. Shared market, initial conditions and environment
- [ ] Write failing tests for common tariff, null limits, 100 kW exports, source-only clearing, unlimited purchases and no terminal restoration.
- [ ] Configure final values; support optional capacity bounds across validation, normalization, projection and environment. ESS peer cap counts load deficits only.
- [ ] Integrate per-departure EV penalty/events and finite scalar trajectory fields; persist session instances and service/end inventory metadata.
- [ ] Version environment/action semantics and reject incompatible old checkpoints through current plugins without changing canonical MATD3.

### 3. MILP parity
- [ ] Write failing optimization tests for unbounded import/pairs with finite derived big-M, prohibited ESS charging peers/export, local-first service, non-cyclic inventory and identical cohort sessions.
- [ ] Implement source/use variables, local-first constraints, session energy variables and departure costs. Keep perfect-foresight/free allocation advantages explicit.
- [ ] Update replay/reference consumers for feasible interval action mapping and session records.

### 4. Verification and results
- [ ] Run targeted tests then `pytest -q`; resolve regressions without weakening the new physical invariants.
- [ ] Run shared pipeline smoke for 1-3 days, both current plugins if feasible, formal test from best_model.pt, all seven coalitions, Equal/Shapley, saved-output plotting.
- [ ] Confirm 15 PNG and 15 PDF, working_runs directory, algorithm-specific YAML, no retained release mutation, no canonical MATD3 diff.
- [ ] Review changes and document checks and residual oracle limitations.

## Review focus

1. Null limits must never put infinities into observations, JSON, tensor normalization or MILP big-M.
2. EV departure exactly at hour boundary, replayed same date across algorithms, hidden day group before arrival.
3. No ESS-to-ESS routing or battery discharge counted as renewable generation/curtailment.
4. No silent loss of EV service penalties when midnight mode is none; end inventories reported even without restoration.
5. Ordinary T and MATD3 share new physical semantics; old checkpoint metadata cannot silently load.

## Execution ledger

- User explicitly requested implementation after iterative design approval. Proceed without an additional permission cycle; writing-plans review gate is superseded by that explicit request and the instruction to finish authorized work.
- Work in the current clean, non-main named checkout so the user receives changes in the requested workspace. No destructive operations, push or release promotion.
- Repository reference `docs/ALGORITHM_DEVELOPMENT.md` is absent; AGENTS.md remains authoritative.
- Numerical parameters not revisited: retain existing MG1/HVAC initial rules and comfort bounds; implement approved EV synthetic values and global price ratio.
