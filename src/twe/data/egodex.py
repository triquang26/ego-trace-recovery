from collections.abc import Iterator
from pathlib import Path

import numpy as np

from twe.data.video import VideoFile
from twe.preprocess.export_windows import Recording


def egodex_instruction(attrs) -> str | None:
    which = int(attrs.get("which_llm_description", 1))
    text = attrs.get("llm_description2" if which == 2 else "llm_description")
    if isinstance(text, bytes):
        text = text.decode()
    return text.strip() if isinstance(text, str) and text.strip() else None


def egodex_episodes(root: Path) -> list[tuple[str, Path, Path]]:
    episodes = []
    for hdf5 in sorted(Path(root).glob("*/*.hdf5")):
        video = hdf5.with_suffix(".mp4")
        if video.exists():
            episodes.append((hdf5.parent.name, hdf5, video))
    return episodes


def caption_key(part: str, task: str, index: str) -> str:
    return f"{part}_{task}_{index}"


def load_captions(path: Path) -> dict[str, str]:
    import json

    return {clip["file"]: clip["caption"] for clip in json.loads(Path(path).read_text()) if clip.get("caption")}


def egodex_recording(part: str, task: str, hdf5: Path, video_path: Path, frame_step: int = 1,
                     caption: str | None = None) -> Recording:
    import h5py

    with h5py.File(hdf5, "r") as handle:
        instruction = egodex_instruction(handle.attrs)
    video = VideoFile(video_path)
    recording_id = f"{part}/{task}/{hdf5.stem}"
    return Recording(recording_id, "egodex", recording_id, instruction, video.timestamps[::frame_step],
                     lambda indices: video.read(np.asarray(indices) * frame_step),
                     {"task": task, "phase": "nominal", "frame_step": frame_step, "motion_caption": caption})


def egodex_recordings(root: Path, start: int = 0, count: int | None = None, frame_step: int = 1,
                      every: int = 1, captions: dict[str, str] | None = None) -> Iterator[Recording]:
    root = Path(root)
    episodes = egodex_episodes(root)
    if captions is not None:
        episodes = [e for e in episodes if caption_key(root.name, e[0], e[1].stem) in captions]
    for task, hdf5, video in episodes[::every][start : None if count is None else start + count]:
        caption = None if captions is None else captions[caption_key(root.name, task, hdf5.stem)]
        yield egodex_recording(root.name, task, hdf5, video, frame_step, caption)


def count_egodex(root: Path, every: int = 1, captions: dict[str, str] | None = None) -> int:
    root = Path(root)
    episodes = egodex_episodes(root)
    if captions is not None:
        episodes = [e for e in episodes if caption_key(root.name, e[0], e[1].stem) in captions]
    return len(episodes[::every])
