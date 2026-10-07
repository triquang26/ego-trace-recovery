# Trace World Expert

Stage 1 của thiết kế ego-robot world model: pretrain trace world expert (DINOv2-Small + T5-small frozen, trace transformer 8×512) bằng flow matching trên B-spline control points của future 3D point tracks.

## Cấu trúc

```text
configs/               world.yaml, stage1.yaml
src/twe/
  config.py            dataclass config, load YAML
  contracts.py         WorldContext / WorldTarget / WorldFeatures, revision strings
  data/                manifest, shards (.npy mmap + meta.jsonl), dataset, mixture sampler, video, EgoDex, synthetic
  preprocess/          letterbox, entity anchors, chunking, camera reference, temporal sampling, B-spline,
                       normalizer, SpaTrackerV2 teacher, window export
  models/              visual/text encoders, trace expert, world module
  training/            objective, pretrain loop, evaluation, checkpoint artifact
  evaluation/          trace metrics, prediction demo, teacher trace video
deploy/modal_app.py    Modal: smoke, normalizer, train, demo
deploy/modal_export.py Modal: tải EgoDex, export teacher targets, teacher video, finalize
tests/                 unit + end-to-end training trên CPU với encoder stub
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
```

Dữ liệu và run nằm trong Modal Volume `trace-world-expert` (`/vol/data/<dataset>`, `/vol/runs/<run>`). Mỗi checkpoint được sync lên `hf://buckets/twanghcmut/trace-world-expert/runs/<run>` (đổi bằng `TWE_HF_BUCKET`). Chạy lại cùng `--run` sẽ resume từ `train_state.pt`.

| Biến | Mặc định | Vai trò |
|---|---|---|
| `TWE_GPU` | `A100-40GB` | GPU train và demo |
| `TWE_EXPORT_GPU` | `A100-80GB` | GPU export teacher |
| `TWE_EXPORT_CONTAINERS` | `4` | Số container export chạy song song |

## Pipeline EgoDex

```bash
modal run --detach deploy/modal_export.py --action download --part test
modal run deploy/modal_export.py --action export --part test --dataset egodex_v2 --episodes 4 --per-shard 4
modal run deploy/modal_export.py --action teacher-videos --dataset egodex_v2
modal run --detach deploy/modal_export.py --action export --part test --dataset egodex_v2 --every 4 --per-shard 10 --val-every 7
modal run deploy/modal_export.py --action finalize --dataset egodex_v2
modal run --detach deploy/modal_app.py --action train --data egodex_v2 --run egodex_v2 --overrides '{"optimizer_updates": 3000}'
modal run deploy/modal_app.py --action demo --data egodex_v2 --run egodex_v2
```

`part=test` vào `validation`, `part1..part5` vào `train`; `--val-every k` tách mỗi shard thứ k thành validation khi chỉ có một part. Shard đã có `entry.json` được bỏ qua khi chạy lại.

## Trích xuất target

- **Anchors**: chỉ từ frame hiện tại. Feature DINO và vị trí được gom thành `anchor_entities` thực thể; mỗi thực thể nhận số điểm theo diện tích mũ `anchor_area_power`, tối thiểu `anchor_min_per_entity`, rải đều bên trong bằng k-means trên toạ độ và bỏ vùng sát mép letterbox. Cùng hàm được dùng ở export và inference.
- **Teacher**: SpaTrackerV2 (commit `7e12274`, camera ước lượng bằng VGGT front). Video được giảm còn 15 fps và chia chunk 10 s; mỗi chunk chạy teacher một lần với query `(t, x, y)` của mọi cửa sổ bên trong. `full_point=True` giữ danh tính query. Reprojection của mỗi query tại frame của nó phải dưới 4 px, nếu không recording bị bỏ.
- **Target**: điểm tương lai đưa về camera tại t, trừ origin, chia median depth hiện tại; sample theo timestamp thật, không nội suy qua gap.
- **Movement filter**: điểm có biên độ chiếu ≥ 35 px trên ảnh 224 được đánh dấu `trace_moving`. Nhãn này dùng cho dynamic metrics, chọn case demo và trọng số loss (`static_point_weight`), không vào input.

## Dataset contract

`manifest.json` liệt kê shards (`pool`, `split`, `count`) cùng `teacher_revision`, `preprocessing_revision`, `coordinate_contract`; loader từ chối revision không khớp. Mỗi shard gồm `rgb`, `image_valid`, `anchor_uv`, `anchor_mask`, `anchor_xyz`, `intrinsics`, `trace`, `trace_valid`, `trace_reliability`, `trace_moving` (`.npy`) và `meta.jsonl`. `normalizer.json` chứa σ theo trục tính trên train split. B-spline target được fit lúc load từ trace và reliability.
