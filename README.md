# Trace World Expert

Stage 1 của thiết kế ego-robot world model: pretrain trace world expert: DINOv2-Base 336px và Grounding DINO tiny (feature ảnh–text đã trộn, frozen) làm encoder, trace transformer 8×512 có history token, head chuyển động cho grounding ngầm bằng flow matching trên B-spline control points của future 3D point tracks.

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
modal run deploy/modal_export.py --action export --part test --dataset egodex_v4 --episodes 4 --per-shard 4
modal run deploy/modal_export.py --action teacher-videos --dataset egodex_v4
modal run --detach deploy/modal_export.py --action export --part test --dataset egodex_v4 --every 1 --per-shard 10 --val-every 7
modal run deploy/modal_export.py --action finalize --dataset egodex_v4
modal run --detach deploy/modal_export.py --action upload --dataset egodex_v4
modal run --detach deploy/modal_app.py --action train --data egodex_v4 --run egodex_v4 --overrides '{"optimizer_updates": 3000}'
modal run deploy/modal_app.py --action demo --data egodex_v4 --run egodex_v4
modal run deploy/modal_app.py --action demo --data egodex_v4 --run egodex_v4 --split train --steps 16
```

`part=test` vào `validation`, `part1..part5` vào `train`; `--val-every k` tách mỗi shard thứ k thành validation khi chỉ có một part. Shard đã có `entry.json` được bỏ qua khi chạy lại.

## Trích xuất target

Theo μ₀ (TraceExtract, Appendix A và `trace_dataset.py` của repo `Yoonkyo/mu0`):

- **Query pool**: mỗi cửa sổ chọn `query_pool` (128) điểm trên frame t − 0.5 s. Tiền cảnh tách bằng thành phần chính của feature DINO; `anchor_foreground_fraction` điểm dành cho tiền cảnh, gom thành `anchor_entities` thực thể, mỗi thực thể nhận điểm theo diện tích mũ `anchor_area_power`, tối thiểu `anchor_min_per_entity`, rải đều bên trong; phần còn lại phủ nền.
- **Teacher**: SpaTrackerV2 (commit `7e12274`, camera ước lượng bằng VGGT front). Video giảm còn 15 fps, chia chunk 10 s; mỗi chunk chạy teacher một lần với query `(t_q, x, y)` của mọi cửa sổ. `full_point=True` giữ danh tính query. Reprojection của mỗi query tại frame của nó phải dưới 4 px, nếu không recording bị bỏ.
- **Lịch sử và tương lai**: điểm được track từ t − 0.5 s qua t tới t + 2 s. Vị trí tại t (chiếu ra `anchor_uv`, phải nằm trong vùng ảnh thật và đủ tin cậy) là gốc; `history` (8 bước) và `trace` (32 bước) là dịch chuyển so với gốc trong camera tại t, chia median depth tại t; sample theo timestamp thật, không nội suy qua gap.
- **Movement filter**: biên độ chiếu tương lai ≥ `moving_threshold_px` (10 px trên ảnh lưu) → `trace_moving`; `finalize` gán lại nhãn theo ngưỡng truyền vào.
- **Train**: như μ₀, mỗi mẫu bốc N ∈ [`min_moving_points`, 64] điểm chuyển động; bỏ lịch sử của cả mẫu với xác suất 0.2 và của từng điểm với xác suất 0.3. Checkpoint tốt nhất theo validation flow lưu ở `world_best.pt`; eval báo cả `ade` và `ade_no_history`.
- **Inference**: `WorldModule.extract_features` chọn 64 anchor từ frame hiện tại bằng cùng bộ chọn thực thể; không có lịch sử thì dùng embedding `no_history`.

## Grounding ngầm

Grounding DINO chỉ dùng làm encoder: token text và feature ảnh đa tỉ lệ sau feature enhancer. Không dùng box. Mỗi điểm nhận feature Grounding DINO tại uv (mức stride 8) và điểm căn chỉnh `max_j ⟨điểm, token_j⟩`; context của trace expert gồm token DINOv2 pooled, token Grounding DINO mức `grounding_context_level` và token text. Train bốc cả điểm chuyển động lẫn đứng yên (`moving_fraction`); head `motion` (BCE, forward riêng tại s = 1) học điểm nào sẽ chuyển động theo câu lệnh, `text_dropout` bỏ câu lệnh ngẫu nhiên.

## Dataset contract

`manifest.json` liệt kê shards (`pool`, `split`, `count`) cùng `teacher_revision`, `preprocessing_revision`, `coordinate_contract`; loader từ chối revision không khớp. Mỗi shard gồm `rgb`, `image_valid`, `anchor_uv`, `anchor_mask`, `anchor_xyz`, `intrinsics`, `trace`, `trace_valid`, `trace_reliability`, `trace_moving`, `history`, `history_valid` (`.npy`, `query_pool` hàng mỗi cửa sổ) và `meta.jsonl`. `normalizer.json` chứa σ theo trục tính trên các điểm chuyển động của train split. B-spline target được fit lúc load từ trace và reliability.
