from pathlib import Path

path = Path("optimization/common_constraints.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one anchor, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


marker = "\ndef solve_dispatch(\n"
helper = '''\ndef _add_shared_pair_capacity_constraints(\n    model: Any,\n    flow_variables: Mapping[tuple[Any, ...], Any],\n    agent_ids: Sequence[str],\n    horizon: int,\n    pair_capacity_kw: float,\n) -> None:\n    \"\"\"Enforce one physical bidirectional capacity for every unordered P2P pair.\"\"\"\n    ids = list(agent_ids)\n    for i, first in enumerate(ids):\n        for second in ids[i + 1:]:\n            for t in range(int(horizon)):\n                model.addConstr(\n                    flow_variables[(first, second, t)]\n                    + flow_variables[(second, first, t)]\n                    <= float(pair_capacity_kw)\n                )\n\n\n'''
replace_once(marker, helper + "def solve_dispatch(\n")

replace_once(
    '''                            ub=pair_capacity_kw, obj=1e-8, type=continuous)\n\n    for t in range(data.horizon):\n''',
    '''                            ub=pair_capacity_kw, obj=1e-8, type=continuous)\n        _add_shared_pair_capacity_constraints(\n            model, variables[\"flow\"], ids, data.horizon, pair_capacity_kw\n        )\n\n    for t in range(data.horizon):\n''',
)

path.write_text(text, encoding="utf-8")
