from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _convert_keys_to_str(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        str_k = str(k)
        if isinstance(v, dict):
            out[str_k] = _convert_keys_to_str(v)
        elif isinstance(v, np.ndarray):
            out[str_k] = v.tolist()
        else:
            out[str_k] = v
    return out


def _restore_int_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            restored_k = int(k)
        except (ValueError, TypeError):
            restored_k = k
        if isinstance(v, dict):
            out[restored_k] = _restore_int_keys(v)
        else:
            out[restored_k] = v
    return out



def save_summary(summary: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(_convert_keys_to_str(summary), f, cls=_NumpyEncoder, indent=2)
    print(f"Saved summary → {p}")


def load_summary(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return _restore_int_keys(raw)


def save_summaries(summaries: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    converted = [_convert_keys_to_str(s) for s in summaries]
    with open(p, "w") as f:
        json.dump(converted, f, cls=_NumpyEncoder, indent=2)
    print(f"Saved {len(summaries)} summaries → {p}")


def load_summaries(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    return [_restore_int_keys(s) for s in raw]