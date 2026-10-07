from pathlib import Path

import av
import numpy as np


class VideoFile:
    def __init__(self, path: Path):
        self.path = str(path)
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
            self.time_base = float(stream.time_base)
        self.pts = np.asarray(pts, dtype=np.int64)
        self.timestamps = (self.pts - self.pts[0]) * self.time_base

    def __len__(self) -> int:
        return len(self.pts)

    def read(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        wanted = {int(self.pts[i]): k for k, i in enumerate(indices)}
        last = int(self.pts[indices.max()])
        out: list[np.ndarray | None] = [None] * len(indices)
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(self.pts[indices.min()]), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts in wanted:
                    out[wanted[frame.pts]] = frame.to_ndarray(format="rgb24")
                if frame.pts >= last:
                    break
        if any(item is None for item in out):
            raise ValueError(f"could not decode all requested frames from {self.path}")
        return np.stack(out)
