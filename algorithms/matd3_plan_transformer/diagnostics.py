from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, ensure_ascii=False,
                                   allow_nan=False, indent=2), encoding='utf-8')


class PlanDiagnosticsRecorder:
    def __init__(self, directory, metadata, actor_contract):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / 'manifest.json'
        contract = json.loads(json.dumps(actor_contract, default=json_value))
        manifest = dict(version='plan-attention-diagnostics-v1', algorithm_slug='matd3_plan_transformer',
            source_run=self.directory.parent.name, actor_contract=contract, metadata=dict(metadata))
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if any(previous.get(k) != manifest[k] for k in ('version','algorithm_slug','source_run','actor_contract')):
                raise ValueError('Diagnostic manifest identity conflict')
            manifest['metadata'] = {**previous['metadata'], **manifest['metadata']}
        checkpoint = manifest['metadata'].get('checkpoint')
        if checkpoint and Path(checkpoint).is_file():
            manifest['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.manifest = manifest
        write_json(path, manifest)
        self.phase = None
        self.phase_metadata = {}
        self.steps = 0

    def _append(self, filename, row):
        with (self.directory / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, default=json_value, ensure_ascii=False, allow_nan=False) + '\n')

    def record_label(self, label):
        self._append('teacher_labels.jsonl', dict(version='cem-feasible-v1', **asdict(label)))

    def record_rejection(self, reason, metadata):
        self._append('teacher_labels.jsonl', dict(valid=False, rejection_reason=reason, metadata=dict(metadata)))

    def set_phase(self, phase, metadata):
        if phase not in (None, 'formal_test'):
            raise ValueError('Unknown diagnostic phase')
        if phase == 'formal_test':
            if self.phase is not None or (self.directory/'formal_test_plan_attention.jsonl').exists():
                raise ValueError('Formal test diagnostic file already exists')
            days, seeds = metadata['days'], metadata['seeds']
            if len(days) != len(seeds) or len(set(days)) != len(days):
                raise ValueError('Invalid day/seed diagnostic identity')
            self.phase_metadata = dict(metadata)
            self.manifest['formal_test'] = dict(metadata)
            write_json(self.directory/'manifest.json', self.manifest)
            self.steps = 0
        elif self.phase is not None:
            write_json(self.directory/'formal_test_diagnostics_summary.json',
                dict(steps=self.steps, rows=self.steps*3, **metadata))
        self.phase = phase

    def record_policy(self, observations, outputs):
        if self.phase != 'formal_test':
            return
        day_index, t = divmod(self.steps, 24)
        if day_index >= len(self.phase_metadata['days']):
            raise ValueError('More policy steps than declared test days')
        day, seed = self.phase_metadata['days'][day_index], self.phase_metadata['seeds'][day_index]
        anchor = datetime.fromisoformat(day).replace(hour=0, minute=0, second=0)
        contract = self.manifest['actor_contract']
        for mg, output in outputs.items():
            schema = contract['observation_schemas'][mg]
            obs = observations[mg]
            actual_t = round(np.arctan2(obs[schema.index('hour_sin')], obs[schema.index('hour_cos')])*24/(2*np.pi)) % 24
            if actual_t != t:
                raise ValueError('Policy observation hour does not match formal test identity')
            pm = output.plan_mask[0].cpu().numpy().astype(bool)
            fm = output.forecast_mask[0].cpu().numpy().astype(bool)
            self._append('formal_test_plan_attention.jsonl', dict(
                source_run=self.manifest['source_run'], checkpoint=self.manifest['metadata'].get('checkpoint'),
                checkpoint_sha256=self.manifest.get('checkpoint_sha256'), day=day, seed=seed, t=t, agent=mg,
                observations=obs, actions=output.actions[0], soc_after=output.soc_after[0],
                ess_net_kw=output.ess_net_kw[0], plan_mask=pm, forecast_mask=fm,
                peer_order=contract['peer_orders'][mg], attention=output.attention[0],
                forecast_relative_hours=list(range(1,25)),
                forecast_token_times=[(anchor+timedelta(hours=t+k)).isoformat() if fm[k-1] else None for k in range(1,25)],
                plan_result_times=[(anchor+timedelta(hours=t+j+1)).isoformat() if pm[j] else None for j in range(24)]))
        self.steps += 1
