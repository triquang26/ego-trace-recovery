# Trace World Expert

Stage 1 của thiết kế ego-robot world model: pretrain trace world expert (DINOv2-Small + T5-small frozen, trace transformer 8×512) bằng flow matching trên B-spline control points của future 3D point tracks.

## Cấu trúc

```text
configs/            world.yaml, stage1.yaml
src/twe/
  config.py         dataclass config, load YAML
  contracts.py      WorldContext / WorldTarget / WorldFeatures, revision strings
  data/             manifest, shard (.npy mmap + meta.jsonl), dataset, mixture sampler, video reader, synthetic
  preprocess/       letterbox, current anchors, camera reference, temporal sampling, B-spline, normalizer, export
  models/           visual/text encoders, trace expert, world module
  training/         objective, pretrain loop, evaluation, checkpoint artifact
  evaluation/       ADE/FDE, dynamic ADE, fit error, validity ECE
deploy/modal_app.py Modal: synthetic smoke, normalizer, train trên A100
deploy/modal_export.py Modal: tải EgoDex, export SpaTrackerV2 targets, finalize manifest + normalizer
tests/              unit + end-to-end training trên CPU với encoder stub
```

## Local

```bash
uv pip install -e ".[dev,deploy]"
pytest
```

## Modal

```bash
modal token set --token-id <id> --token-secret <secret> --profile=<workspace>
modal profile activate <workspace>
modal secret create huggingface HF_TOKEN=<hf_token>
modal run deploy/modal_app.py --action smoke
modal run deploy/modal_app.py --action normalizer --data <dataset>
modal run --detach deploy/modal_app.py --action train --data <dataset> --run <run_name>
```

GPU mặc định `A100-40GB` (đổi bằng `TWE_GPU`). Dữ liệu và run nằm trong Modal Volume `trace-world-expert` (`/vol/data/<dataset>`, `/vol/runs/<run>`). Mỗi checkpoint được sync lên `hf://buckets/twanghcmut/trace-world-expert/runs/<run>` (đổi bằng `TWE_HF_BUCKET`). Chạy lại cùng `--run` sẽ resume từ `train_state.pt`.

## Export EgoDex

```bash
modal run --detach deploy/modal_export.py --action download --part test
modal run deploy/modal_export.py --action export --part test --dataset egodex_v1 --episodes 4
modal run --detach deploy/modal_export.py --action export --part test --dataset egodex_v1
modal run deploy/modal_export.py --action finalize --dataset egodex_v1
modal run --detach deploy/modal_app.py --action train --data egodex_v1 --run egodex_v1
```

`part=test` vào split `validation`, `part1..part5` vào `train`. Mỗi job GPU export `--per-shard` episode thành một shard, cửa sổ cách nhau `--stride` giây. SpaTrackerV2 pin commit `7e12274`; teacher kiểm tra reprojection của query tại frame hiện tại để xác nhận convention OpenCV trước khi ghi target.

## Dataset contract

`/vol/data/<dataset>/manifest.json` liệt kê shards (`pool`, `split`, `count`) cùng `teacher_revision`, `preprocessing_revision`, `coordinate_contract`; loader từ chối revision không khớp. Mỗi shard gồm `rgb`, `image_valid`, `anchor_uv`, `anchor_mask`, `trace`, `trace_valid`, `trace_reliability` (`.npy`) và `meta.jsonl`. `trace` là relative displacement trong camera tại t, chia cho median depth hiện tại; `normalizer.json` chứa σ theo trục tính trên train split. B-spline target được fit lúc load từ trace và reliability.

`preprocess/export_windows.py` tạo shard từ `Recording` + một `TrackTeacher` (`preprocess/teacher.py`). Anchors chọn từ ảnh hiện tại, teacher query đúng các anchor đó.
