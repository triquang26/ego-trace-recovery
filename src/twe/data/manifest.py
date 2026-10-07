import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from twe.contracts import COORDINATE_CONTRACT, POOLS, PREPROCESSING_REVISION


@dataclass
class ShardEntry:
    path: str
    pool: str
    split: str
    count: int

    def __post_init__(self) -> None:
        if self.pool not in POOLS:
            raise ValueError(f"unknown pool {self.pool}")


@dataclass
class Manifest:
    shards: list[ShardEntry]
    teacher_revision: str
    preprocessing_revision: str = PREPROCESSING_REVISION
    coordinate_contract: str = COORDINATE_CONTRACT
    sources: dict = field(default_factory=dict)

    def select(self, split: str) -> list[ShardEntry]:
        return [entry for entry in self.shards if entry.split == split]

    def write(self, root: Path) -> None:
        record = {**asdict(self), "shards": [asdict(e) for e in self.shards]}
        (Path(root) / "manifest.json").write_text(json.dumps(record, indent=2))

    @classmethod
    def read(cls, root: Path) -> "Manifest":
        record = json.loads((Path(root) / "manifest.json").read_text())
        manifest = cls(**{**record, "shards": [ShardEntry(**e) for e in record["shards"]]})
        if manifest.preprocessing_revision != PREPROCESSING_REVISION:
            raise ValueError(f"preprocessing revision {manifest.preprocessing_revision} != {PREPROCESSING_REVISION}")
        if manifest.coordinate_contract != COORDINATE_CONTRACT:
            raise ValueError(f"coordinate contract {manifest.coordinate_contract} != {COORDINATE_CONTRACT}")
        return manifest
