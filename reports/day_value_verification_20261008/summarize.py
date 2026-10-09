"""Build a compact evidence note and an inline sources receipt from verified rows."""
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
s=json.loads((OUT/'verification_summary.json').read_text())
c=json.loads((OUT/'independent_checks.json').read_text())
f=pd.read_csv(OUT/'ess_action_pairs.csv')
assert len(f)==271 and not f.duplicated(['date','hour','agent','candidate_raw']).any()
assert (f.filter(regex='^delta_cost_').sum(axis=1)+f.cost_reduction).abs().max()<1e-7
case=next(x for x in c['biggest_missed_ess'] if x['date']=='2018-05-30')
mode_names={'ess_free':'储能','hvac_free':'空调','ev_free':'EV','mill_free':'磨机'}
opportunities=[dict(module=mode_names[x['mode']],daily_cost_reduction=x['mean_saving'])
               for x in s['frozen_curve_reoptimization'] if x['mode'] in mode_names]
gap=s['formal_cost_mean']-s['generalized_reference_mean']
note=f'''# 日内动作价值验证：2026-10-08

判断：完整 Dueling 结果中存在动作累计价值误判，包含当前成本与后续成本方向相反时的错误排序，值得进行统一改进研究。不能将全部问题归因为奖励稀疏，也未证实时间信用分配是唯一训练根因。

普通 MATD3 本地运行尚未完成；本结论针对 Dueling 5000 轮完整训练、第 4600 轮最优检查点及其 55 天正式测试。没有更新模型参数，没有修改算法、环境、奖励或正式结果。

## 原始结果复核

- 核对冻结检查点与正式 best_model.pt 哈希一致，并核验既有诊断的算法、环境、数据及正式逐时文件哈希。
- 重放 55 天保存的原始动作，与正式日成本最大差 {s['saved_action_replay_max_cost_error']:.3g}。
- 重新生成策略动作，与正式日成本最大差 {s['baseline_replay_max_cost_error']:.3g}；这是远小于 0.1 成本判别阈值的数值误差。
- 成本采用环境广义成本，包括购电及碳、峰值、舒适等惩罚，不等同于现金利润。

## 新增储能诊断

在每月测试日期中预先选取居中日期，共 12 日，每日检查 0、6、12、18 时。候选为原始 ESS 动作 ±0.25 和 0，裁剪并排除实际功率不变的情况。每次只修改一个智能体当前的储能动作；此后所有智能体按原策略、相同外生场景运行到日末。

- 有效比较 271 对，其中全天成本差绝对值大于 0.1 的比较 158 对。
- Q1 排反 68/158，Q2 排反 62/158。这是抽样动作对的比例，不能解释为所有运行时动作的错误概率。
- 当前小时成本变化方向与全天变化方向相反的比较有 56 对，其中 Q1 排反 34 对。
- 42 个实际更省钱的候选被 Q1 评为更差，覆盖 10/12 个抽样日期。

最直接的跨时段例子：2018-05-30 06:00，mg2 储能放电由 {case['reference_kw']:.2f} kW 降至 {case['candidate_kw']:.2f} kW。当前成本增加 {-case['immediate_cost_reduction']:.3f}，后续成本降低 {case['later_cost_reduction']:.3f}，全天净降低 {case['cost_reduction']:.3f}；Q1、Q2 反而预测更差。磨机、EV、未供电和舒适成本均没有增加。

## 成本差能说明什么

每个抽样日最多选择一次被 Q1 误判的省钱储能动作，不把重叠候选收益相加。12 日平均降低 {s['ess_one_correction_per_sampled_day_mean']:.3f}，中位数 {c['ess_one_correction_median']:.3f}；最大日期降低 33.034，剔除它后其余 11 日平均 {c['ess_one_correction_mean_excluding_largest']:.3f}。若额外要求磨机、EV、未供电、舒适的成本均不增加，平均仍可降低 {c['ess_best_no_extra_any_service_cost_mean']:.3f}。

这个数字使用实际未来回放来挑选候选，证明存在可避免的成本机会，不是部署策略或重新训练后收益预测。没有证据将 12 日均值外推到所有日期。

已核验的 55 日完整调度机会：仅重排储能约 52.83/日，空调约 46.71/日，EV 约 29.76/日，磨机约 24.67/日。其余模块实际功率曲线固定，并采用原环境回放的成本，而非直接采用不可完全执行的 EV MILP 成本。各模块不能相加。这些是利用真实未来信息的调度机会，不能全部归因于 Critic 误判。

原策略日均成本 {s['formal_cost_mean']:.2f}，保存的全信息参考 {s['generalized_reference_mean']:.2f}，差 {gap:.2f}，约 {100*gap/s['formal_cost_mean']:.2f}%；这个总差距同样不是价值学习改进的保证收益。

## 命名与原因边界

环境每小时提供成本奖励，因此不属于整体奖励稀疏；EV 离站等个别奖励项确有事件性/延迟性。前面动作影响后续收益的归因难点可以称为时间信用分配，团队共享奖励还涉及智能体信用分配。后者是否为主要瓶颈没有独立验证。

更稳妥的研究表述：多智能体日内调度中，跨时段动作贡献难以准确评估，导致长期动作价值估计与实际累计成本变化不一致。与此同时，日末 HVAC 在无未来状态时也有误判，说明即时动作成本映射同样有问题；不能仅以时间信用分配解释全部错误。

没有保存训练 replay/每步优化历史，现有诊断不区分探索覆盖、优化过程和网络表达的因果占比。

## 证据文件

- `verify.py`：冻结策略、55 日重放、271 对 ESS 动作及五个 DR 案例复现。
- `ess_action_pairs.csv`：实际累计/即时/后续成本、双 Q 排序、九项成本变化。
- `policy_replay.csv`、`reproduced_dr_cases.csv`：原结果一致性与 DR 案例。
- `verification_summary.json`、`independent_checks.json`、`sources.json`：统计、独立账目复核与来源哈希。
- 原有全日调度机会证据：`reports/milp_gap_audit_20261008/`；DR 误差定位：`reports/critic_error_location_20261008/`。

可复现：在 MKL_THREADING_LAYER=SEQUENTIAL、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1 的环境下执行 `python reports/day_value_verification_20261008/verify.py`。仅写诊断目录，不修改原始运行。
'''
(OUT/'README.md').write_text(note,encoding='utf-8')
common=['Dueling 完整训练 5000 轮；正式最优模型来自第 4600 轮','广义成本包含非现金惩罚；普通 MATD3 尚未完成']
receipt={'schemaVersion':1,'items':[
 {'id':'action-value-errors','title':'实际跨时段成本变化与 Critic 动作排序不一致','queries':[
  {'id':'ess-paired-rollouts','source':{'label':'冻结最优模型的储能反事实回放',
    'files':[{'label':'day_value_verification_20261008/ess_action_pairs.csv'},{'label':'day_value_verification_20261008/verify.py'}],
    'metricDefinitions':[{'label':'动作排序错误','definition':'全天成本差绝对值大于 0.1，且 Q1 动作差与实际累计奖励差符号相反。'}],
    'filters':common+['每月预选一个测试日期，共 12 日；检查 0、6、12、18 时'],
    'caveats':['抽样动作对比例不是所有动作的错误概率；误判不确定唯一训练根因。'],
    'evidenceFlow':[{'kind':'source','title':'原结果一致性','detail':f"55 天保存动作重放成本最大误差 {s['saved_action_replay_max_cost_error']:.3g}，冻结与正式检查点哈希一致。"}]},
   'rows':[{'comparison':'ESS meaningful pairs','count':158},{'comparison':'Q1 wrong','count':68},
           {'comparison':'current-versus-day sign reversal','count':56},{'comparison':'Q1 wrong among reversals','count':34}],
   'methods':[{'language':'calculation','code':'2018-05-30 06:00: current cost +9.725; later cost -14.631696; whole-day cost -4.906696.\nOne current action changed; all later actions follow the same frozen policy.'}]},
  {'id':'dr-reproduction','source':{'label':'五个 DR 案例重新回放','files':[{'label':'day_value_verification_20261008/reproduced_dr_cases.csv'}],
    'filters':common,'caveats':['案例刻意选取以验证现象，不能用于总体误判率估计。']},'rows':s['reproduced_dr_cases']}]},
 {'id':'cost-opportunities','title':'误判动作有实际成本机会，整体调度仍有更大空间','queries':[
  {'id':'one-action','source':{'label':'12 个预选日期的单次 ESS 修正','files':[{'label':'day_value_verification_20261008/verification_summary.json'},{'label':'day_value_verification_20261008/independent_checks.json'}],
    'filters':common+['每日期至多一次被 Q1 误判的省钱动作'],
    'caveats':['使用未来回放挑选候选，是诊断机会，不是重新训练收益；均值受一个 33.034 日期影响。']},
   'rows':[{'measure':'12-day mean','saving':s['ess_one_correction_per_sampled_day_mean']},{'measure':'12-day median','saving':c['ess_one_correction_median']},
           {'measure':'mean without any extra service cost','saving':c['ess_best_no_extra_any_service_cost_mean']}],
   'methods':[{'language':'calculation','code':'Per date: max(cost reduction among tested positive-saving, negative-Q1-change candidates), or 0 if none.\nMean/median over all 12 preselected dates; candidates from one date are never added.'}]},
  {'id':'full-day-space','source':{'label':'55 日固定其他功率曲线的全日重排','files':[{'label':'milp_gap_audit_20261008/environment_replay_results.csv'},{'label':'test_daily_metrics.csv'}],
    'filters':common+['同一组 55 个正式测试日期'],
    'caveats':['优化使用完整未来信息，不能全归因于 Critic；各模块改善有交互，不能相加。']},
   'rows':opportunities+[{'module':'full-information saved-reference gap','daily_cost_reduction':gap}],
   'methods':[{'language':'calculation','code':f"Mean policy cost {s['formal_cost_mean']:.9f} - mean generalized reference {s['generalized_reference_mean']:.9f} = {gap:.9f}.\nEach module saving is the mean of policy cost minus unchanged-environment replay cost over the same 55 dates."}]}]}
]}
for item in receipt['items']:
    for query in item['queries']:
        query['columns']=list(query['rows'][0])
(OUT/'sources_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('Evidence note and two-card sources receipt written; 271 paired rows and cost identities verified.')
