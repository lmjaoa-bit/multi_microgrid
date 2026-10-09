"""Size observed one-action opportunities and explicit gap-recovery scenarios.

No environment rollouts, training updates, or experimental configuration changes.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/DUELINGCRITICT-QH24_2026-10-07_22-11-08_seed42/artifacts'
PAIR = HERE / 'latest/pair_decomposition.csv'
DAILY = RUN / 'test_daily_metrics.csv'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    pairs, daily = read_csv(PAIR), read_csv(DAILY)
    assert len(daily) == 55 and len({r['date'] for r in daily}) == 55
    summary = json.loads((RUN / 'test_summary.json').read_text(encoding='utf-8'))
    cost = mean(float(r['algorithm_generalized_system_cost']) for r in daily)
    reference = mean(float(r['generalized_reference_cost']) for r in daily)
    assert abs(cost - summary['generalized_system_cost_mean']) < 1e-9
    assert abs(reference - summary['generalized_reference_cost_mean']) < 1e-9
    for r in daily:
        assert abs(float(r['algorithm_generalized_system_cost']) + 100 * float(r['algorithm_team_reward'])) < 1e-8
    for r in pairs:
        delta_cost = sum(float(v) for k, v in r.items() if k.startswith('delta_cost_'))
        assert abs(delta_cost + 100 * float(r['gain'])) < 1e-8
    opportunities = []
    for agent, device in [('mg1', 'Mill'), ('mg2', 'HVAC'), ('mg3', 'EV')]:
        local = [r for r in pairs if r['agent'] == agent and r['sample'] == 'stratified'
                 and abs(abs(float(r['power_delta_kw'])) - 1) < 1e-5]
        missed = [r for r in local if float(r['gain']) > .001 and float(r['dq1']) < 0]
        values = [100 * float(r['gain']) for r in missed]
        opportunities.append(dict(device=device, pairs=len(local), missed_positive=len(missed),
                                  median_cost_reduction=median(values),
                                  min_cost_reduction=min(values), max_cost_reduction=max(values)))
    scenarios = [dict(gap_recovery_fraction=f, daily_cost_reduction=(cost-reference)*f,
                      cost_reduction_pct=100*(cost-reference)*f/cost) for f in [.1, .25, .5]]
    result = dict(checkpoint_episode=4600, days=55, generalized_cost_mean=cost,
                  generalized_reference_mean=reference, broader_reference_gap=cost-reference,
                  broader_reference_gap_pct=100*(cost-reference)/cost,
                  single_action_opportunities=opportunities, scenarios=scenarios,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [PAIR, DAILY]},
                  checks=['55 unique dates', 'Daily means reconcile to formal summary',
                          'Every daily reward reconciles to cost / 100',
                          'Every paired reward gain reconciles to saved cost components'],
                  caveats=['Scenarios are assumptions, not forecasts or confidence intervals.',
                           'Reference allocation differs from online clearing; the gap is not an achievable Critic-only bound.',
                           'Single-action counterfactual gains overlap and must not be summed over a day.',
                           'Generalized cost contains noncash penalties and is not accounting profit.',
                           'No retrained policy has been evaluated.'])
    (HERE / 'benefit_estimate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    receipt = {'schemaVersion': 1, 'items': [
        {'id': 'local-opportunities', 'title': 'Observed gains from individual misranked DR actions', 'queries': [
            {'id': 'local-gains', 'source': {'label': 'Final-checkpoint paired action audit',
              'files': [{'label': 'latest/pair_decomposition.csv'}, {'label': 'estimate_benefit.py'}],
              'metricDefinitions': [{'label': 'Single-action generalized cost reduction',
                'definition': 'A first-step effective action changes by 1 kW; subsequent steps use the unchanged deterministic policy. Reward gain is converted with scale 100.'}],
              'filters': ['Episode 4600 best checkpoint', 'Stratified audit states', 'True gain > 0.001 and Q1 delta < 0'],
              'caveats': result['caveats'][2:]}, 'rows': opportunities,
             'columns': list(opportunities[0]), 'preview': {'kind': 'aggregate', 'note': 'Three device aggregates.'},
             'evidenceFlow': [{'kind': 'validation', 'title': 'Cost identity', 'text': 'All 326 pair rows reconcile reward gains to saved cost-component deltas within 1e-8.'}]}]},
        {'id': 'recovery-scenarios', 'title': 'Hypothetical recovery of part of the broader reference gap', 'queries': [
            {'id': 'cost-gap', 'source': {'label': '55-day formal test',
              'files': [{'label': 'test_daily_metrics.csv'}, {'label': 'test_summary.json'}, {'label': 'estimate_benefit.py'}],
              'metricDefinitions': [{'label': 'Baseline generalized daily cost', 'definition': 'Mean generalized system cost across the 55 formal-test dates.'},
                                    {'label': 'Recovery scenario', 'definition': 'Assumed fraction of the mean algorithm-minus-reference cost gap recovered, divided by baseline mean cost.'}],
              'caveats': result['caveats']},
             'rows': [dict(baseline=cost, reference=reference, gap=cost-reference, **r) for r in scenarios],
             'columns': ['baseline', 'reference', 'gap', 'gap_recovery_fraction', 'daily_cost_reduction', 'cost_reduction_pct'],
             'preview': {'kind': 'aggregate', 'note': 'Illustrative scenarios based on observed baseline values.'},
             'methods': [{'language': 'calculation', 'code': 'gap = mean(algorithm_generalized_system_cost) - mean(generalized_reference_cost)\nsaving = gap * assumed_recovery_fraction\npercent = 100 * saving / mean(algorithm_generalized_system_cost)'}],
             'evidenceFlow': [{'kind': 'validation', 'title': 'Baseline reconciliation', 'text': '55 unique dates; both recomputed means match the formal summary within 1e-9. All daily reward/cost identities match within 1e-8.'}]}]}
    ]}
    (HERE / 'benefit_sources.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
