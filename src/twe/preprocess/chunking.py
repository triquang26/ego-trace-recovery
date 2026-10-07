from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    frames: np.ndarray
    starts: list[int]


def window_starts(timestamps: np.ndarray, horizon: float, stride: float) -> list[int]:
    starts, next_time = [], float(timestamps[0])
    last_start = float(timestamps[-1]) - horizon
    for index, time in enumerate(timestamps):
        if time > last_start:
            break
        if time >= next_time:
            starts.append(index)
            next_time = time + stride
    return starts


def plan_chunks(timestamps: np.ndarray, starts: list[int], span: float, chunk_seconds: float) -> list[Chunk]:
    if chunk_seconds < span:
        raise ValueError("chunk must cover at least one window")
    chunks, pending = [], list(starts)
    while pending:
        first = pending[0]
        begin = float(timestamps[first])
        members = [s for s in pending if timestamps[s] + span <= begin + chunk_seconds]
        last_time = float(timestamps[members[-1]]) + span
        stop = int(np.searchsorted(timestamps, last_time, side="right"))
        frames = np.arange(first, stop)
        chunks.append(Chunk(frames, [s - first for s in members]))
        pending = pending[len(members):]
    return chunks
