"""ONNX export of the streaming (single-step) decoders and CPU latency measurement."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def export_onnx(step_module: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...],
                input_names: list[str], output_names: list[str], path: str | Path, opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(step_module.eval(), example_inputs, str(path), input_names=input_names,
                      output_names=output_names, opset_version=opset, dynamo=False)
    return path


def onnx_session(path: str | Path, threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def stream_onnx(session, features: np.ndarray, state: dict[str, np.ndarray],
                state_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Feed a feature sequence one step at a time, carrying the state between calls.

    ``state_map`` maps each state output name to the input name it feeds next step.
    Returns the predictions and the wall-clock latency of every call (ms).
    """
    outputs, latencies = [], []
    feeds = dict(state)
    names = [o.name for o in session.get_outputs()]
    for step in features.astype(np.float32):
        feeds["x"] = step[None]
        t0 = time.perf_counter()
        result = dict(zip(names, session.run(None, feeds)))
        latencies.append((time.perf_counter() - t0) * 1e3)
        outputs.append(result["y"][0])
        for out_name, in_name in state_map.items():
            feeds[in_name] = result[out_name]
    return np.asarray(outputs), np.asarray(latencies)


def latency_summary(latencies_ms: np.ndarray, warmup: int = 100) -> dict[str, float]:
    lat = latencies_ms[warmup:]
    return {"median_ms": float(np.median(lat)), "p99_ms": float(np.percentile(lat, 99)),
            "max_ms": float(lat.max()), "n_steps": int(lat.size)}
