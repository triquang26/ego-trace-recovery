# Trace World Expert

Stage 1 của thiết kế ego-robot world model: pretrain trace world expert: DINOv2-Base hoặc DINOv3 ViT-B/16 336px và Grounding DINO tiny (frozen) làm encoder, 4 lớp fusion ảnh–text hai chiều khởi tạo từ Grounding DINO (train), trace transformer 8×512 có history token, head chuyển động cho grounding ngầm bằng flow matching trên B-spline control points của future 3D point tracks.

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

## Đánh giá tổng quát hoá

```bash
modal run deploy/modal_tools.py --action fetch-mu0
modal run deploy/modal_app.py --action benchmark --data egodex_v4 --run scale_f100,scale_f067
modal run deploy/modal_tools.py --action prompts --run scale_f100
modal run deploy/modal_tools.py --action compare-teacher --group test_dataset_egodex
```

- `heldout_tasks` trong `configs/stage1.yaml` là các task EgoDex không bao giờ vào train hay validation; `train_shard_fraction` lấy một phần shard train (cố định theo hash tên shard) để đo đường scaling.
- `benchmark` ghi `runs/<run>/benchmark.json`: trên điểm chuyển động, ADE một mẫu, trung bình 5 mẫu, minADE@5, baseline đứng yên, ADE khi bỏ history (đơn vị % cạnh ảnh 336), và AUROC của head chuyển động. Nhóm: validation task đã thấy, task giữ riêng, và test set μ₀ (TraceExtract: DROID, Franka, UR3, VIMA, EgoDex, video người và robot), quy về cùng (Δu, Δv, Δlog z) và horizon 2 s (`src/twe/data/traceextract.py`; DROID 5 Hz, còn lại 10 Hz).
- `prompts` vẽ cùng cảnh với nhiều câu lệnh, có classifier-free guidance `w` (`WorldModule.sample_controls(..., guidance, null_inputs)`).
- `compare-teacher` chạy teacher SpaTrackerV2 trên đúng frame và keypoint của TraceExtract, ghi lệch vị trí, hướng, độ rung, tỉ lệ depth và ảnh đặt cạnh nhau vào `/vol/viz/teacher_compare`.

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

## Fusion ảnh–text

Như TurboVLA: token DINO (24×24 với DINOv2, 21×21 với DINOv3) qua `VisionProjection` về 256 chiều cộng sincos 2D; text là output BERT + `text_projection` của Grounding DINO (trước enhancer, mask sub-sentence gốc). `fusion_layers` lớp đầu, mỗi lớp gồm bi-attention (`fusion_layer`) và text self-attention (`text_enhancer_layer`), được sao chép nguyên trọng số từ encoder Grounding DINO rồi train; bỏ deformable attention của ảnh. Ảnh đã trộn được pool về `pooled_grid` làm context, lấy mẫu tại uv cộng vào token mỗi điểm; text đã trộn thay text context. `fusion_layers: 0` trả về kiến trúc cũ. `ema_decay` giữ trung bình trượt của tham số train, dùng cho eval và checkpoint. Eval báo thêm `flow_null_text`, `flow_shuffled_text`: flow loss khi bỏ câu lệnh hoặc đổi câu lệnh giữa các window; cao hơn `flow` nghĩa là model dùng câu lệnh.

## Dataset contract

`manifest.json` liệt kê shards (`pool`, `split`, `count`) cùng `teacher_revision`, `preprocessing_revision`, `coordinate_contract`; loader từ chối revision không khớp. Mỗi shard gồm `rgb`, `image_valid`, `anchor_uv`, `anchor_mask`, `anchor_xyz`, `intrinsics`, `trace`, `trace_valid`, `trace_reliability`, `trace_moving`, `history`, `history_valid` (`.npy`, `query_pool` hàng mỗi cửa sổ) và `meta.jsonl`. `normalizer.json` chứa σ theo trục tính trên các điểm chuyển động của train split. B-spline target được fit lúc load từ trace và reliability.
