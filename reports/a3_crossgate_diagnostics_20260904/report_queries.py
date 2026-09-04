"""Recompute report chart datasets directly from saved CSV/JSON via SQLite.

Input sensitivity/gate results remain Python/PyTorch diagnostics in evidence.json.
The SQL here independently reconciles displayed accounting charts with that file.
"""
from pathlib import Path
import json
import sqlite3
import pandas as pd

OUT=Path(__file__).resolve().parent
e=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
db=sqlite3.connect(OUT/'report_sources.sqlite')
hourly=[];daily=[];validation=[]
for slug,r in e['runs'].items():
    label='CrossGate' if slug.startswith('cross') else '普通 MATD3'
    p=Path(r['path'])
    h=pd.concat([pd.read_csv(f) for f in sorted((p/'test_hourly_trajectories').glob('*.csv'))])
    for mg in ['mg1','mg2','mg3']:
        fields={c:c[len(mg)+1:] for c in h.columns if c.startswith(mg+'_')}
        part=h[['date','hour',*fields]].rename(columns=fields)
        hourly.append(part.assign(algorithm=label,microgrid=mg.upper()))
    daily.append(pd.read_csv(p/'test_daily_metrics.csv').assign(algorithm=label))
    validation.append(pd.DataFrame(r['validation']).assign(algorithm=label))
pd.concat(hourly).to_sql('hourly_saved',db,if_exists='replace',index=False)
pd.concat(daily).to_sql('daily_saved',db,if_exists='replace',index=False)
pd.concat(validation).to_sql('validation_saved',db,if_exists='replace',index=False)
queries={
'microgrids': '''SELECT microgrid,
 SUM(CASE WHEN algorithm='CrossGate' THEN 1 ELSE -1 END *
 (hourly_profit-p2p_sale_revenue+p2p_purchase_cost))/55.0 AS external_gain,
 SUM(CASE WHEN algorithm='CrossGate' THEN 1 ELSE -1 END * hourly_profit)/55.0 AS physical_gain,
 SUM(CASE WHEN algorithm='CrossGate' THEN 1 ELSE -1 END *
 (p2p_sale_revenue-p2p_purchase_cost))/55.0 AS internal_transfer,
 COUNT(DISTINCT date) AS test_days
 FROM hourly_saved GROUP BY microgrid ORDER BY microgrid''',
'economics': '''WITH totals AS (
 SELECT algorithm, SUM(grid_import_cost)/55.0 AS imports,
 SUM(grid_export_revenue)/55.0 AS exports,
 SUM(battery_degradation_cost)/55.0 AS degradation,
 SUM(grid_export_revenue-grid_import_cost+p2p_sale_revenue-p2p_purchase_cost
     -battery_degradation_cost-hourly_profit)/55.0 AS other
 FROM hourly_saved GROUP BY algorithm
 ), pair AS (SELECT p.*,c.imports AS ci,c.exports AS ce,c.degradation AS cd,c.other AS co
 FROM totals p CROSS JOIN totals c WHERE p.algorithm='普通 MATD3' AND c.algorithm='CrossGate')
 SELECT '购电成本减少' AS component, imports-ci AS gain, imports AS plain, ci AS crossgate,55 AS test_days FROM pair
 UNION ALL SELECT '外送收入减少',ce-exports,exports,ce,55 FROM pair
 UNION ALL SELECT '电池损耗成本减少',degradation-cd,degradation,cd,55 FROM pair
 UNION ALL SELECT '其他成本减少',other-co,other,co,55 FROM pair''',
'terminal': '''SELECT algorithm,AVG(CASE WHEN hour=22 THEN soc END) AS soc22,
 AVG(CASE WHEN hour=23 THEN battery_charge_kw END) AS charge23,
 AVG(CASE WHEN hour=23 THEN grid_import_cost END) AS cost23,
 AVG(CASE WHEN hour=23 THEN dispatch_forced END) AS forced
 FROM hourly_saved WHERE microgrid='MG3' AND hour IN (22,23)
 GROUP BY algorithm ORDER BY algorithm''',
'monthly': '''WITH pair AS (
 SELECT c.date,c.algorithm_team_profit-p.algorithm_team_profit AS gain
 FROM daily_saved c JOIN daily_saved p ON c.date=p.date AND c.test_seed=p.test_seed
 WHERE c.algorithm='CrossGate' AND p.algorithm='普通 MATD3')
 SELECT SUBSTR(date,1,7) AS month,AVG(gain) AS gain,COUNT(*) AS count,SUM(gain) AS total
 FROM pair GROUP BY SUBSTR(date,1,7) ORDER BY month''',
'validation': '''SELECT episode,profit,algorithm FROM validation_saved ORDER BY episode,algorithm''',
}
datasets={k:pd.read_sql_query(q,db).to_dict('records') for k,q in queries.items()}
assert abs(sum(r['gain'] for r in datasets['economics'])-e['paired']['mean'])<1e-7
assert abs(sum(r['external_gain'] for r in datasets['microgrids'])-e['paired']['mean'])<1e-7
assert abs(sum(r['total'] for r in datasets['monthly'])-e['paired']['total'])<1e-7
(OUT/'report_queries.json').write_text(json.dumps(dict(queries=queries,datasets=datasets),ensure_ascii=False,indent=2),encoding='utf-8')
db.close()
print('SQL chart datasets independently reconciled with saved-result analysis.')
