# Task 1 实施报告

## 范围

完成可执行的 Characteristic Value 归一化与 exact Shapley 语义；未实现 Task 2 及以后内容，未修改 canonical `algorithms/matd3/`。

## TDD 记录

1. 先新增 `tests/test_matd3_shapley_q.py` 中的四个数学测试。
2. 按简报运行目标测试：

   `pytest -q tests/test_matd3_shapley_q.py::test_characteristic_normalization_uses_nonzero_independent_value`

   结果为 RED：收集阶段 `ModuleNotFoundError: No module named 'algorithms.matd3_shapley_q.coalition'`。
3. 执行强制脚手架命令：

   `python scripts/create_algorithm_variant.py --name matd3_shapley_q`

   测试导入产生的空 `__pycache__` 目录导致第一次执行报告目录已存在；确认目录无源文件后清理该缓存并重跑，脚手架成功生成包与独立 YAML。
4. 在脚手架上实现 `coalition.py` 与必要包导出。
5. 聚焦测试：

   `pytest -q tests/test_matd3_shapley_q.py -k "coalition or shapley or characteristic"`

   结果为 GREEN：`4 passed`。

6. 完整回归：

   `pytest -q`

   结果：`88 passed, 2 warnings`。警告为既有 JSON 非有限值替换与非正 Shapley gain 场景的运行时警告。

## 实现内容

- `all_coalitions` 生成空集至全集的全部 coalition，并拒绝重复 agent ID。
- `normalize_characteristic_values` 严格校验 mapping 恰好包含全部 coalition，以真实 `J(empty)` 为 baseline；因此空集归一化为 0，全集为合作 surplus。
- `enumerate_predecessor_coalitions` 与 `shapley_weight` 提供标准 predecessor marginal formula 所需辅助逻辑。
- `exact_shapley_values` 基于归一化值计算 exact Shapley，并满足效率性与零合作变化语义。

## 自审与提交

- `git diff --check` 与 staged diff check 均通过。
- canonical `algorithms/matd3/` 无差异。
- 已提交全部 Task 1 变更，commit：`27c9177` (`feat: normalize Shapley values against independent operation`)。

顾虑：无。
