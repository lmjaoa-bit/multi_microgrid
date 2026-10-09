import dataclasses
import json
from pathlib import Path

import numpy as np


def write_json(path, payload):
    def default(value):
        if dataclasses.is_dataclass(value):
            return {field.name:getattr(value,field.name) for field in dataclasses.fields(value)}
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,default=default,indent=2,allow_nan=False),encoding='utf-8')
