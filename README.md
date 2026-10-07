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

## Dataset contract

`/vol/data/<dataset>/manifest.json` liệt kê shards (`pool`, `split`, `count`) cùng `teacher_revision`, `preprocessing_revision`, `coordinate_contract`; loader từ chối revision không khớp. Mỗi shard gồm `rgb`, `image_valid`, `anchor_uv`, `anchor_mask`, `trace`, `trace_valid`, `trace_reliability` (`.npy`) và `meta.jsonl`. `trace` là relative displacement trong camera tại t, chia cho median depth hiện tại; `normalizer.json` chứa σ theo trục tính trên train split. B-spline target được fit lúc load từ trace và reliability.

`preprocess/export_windows.py` tạo shard từ `Recording` + một `TrackTeacher` (`preprocess/teacher.py`). Anchors chọn từ ảnh hiện tại, teacher query đúng các anchor đó.
