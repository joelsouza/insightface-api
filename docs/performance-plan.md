# Performance plan: InsightFace API on 4 CPU / 16 GB (Coolify)

## Context

The API runs one Uvicorn process with a 4-thread pool for inference. Traffic is **batch jobs from object storage** (bursts of many requests). The audit found that the process wastes CPU on thread oversubscription, blocks the event loop with image decode, has no back-pressure under bursts, and downloads 280 MB of models on every cold start. The user approved: rounding embeddings to 6 decimals, baking model files into the image, and adding an `image_url` input.

Goal: higher throughput per CPU, stable latency under bursts, fast cold start. No change to the Flask dev app (`src/app.py`, `src/api/routes.py`) beyond keeping shared helpers compatible.

## Findings (ranked by impact)

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | **ONNX thread oversubscription.** Each ORT session defaults to `intra_op_num_threads = physical cores` (4). With 4 pool workers that is up to 16 busy threads on 4 CPUs. Intra-op threads also spin-wait by default and steal CPU from other sessions. `FaceAnalysis` gives no way to pass `SessionOptions` (`model_zoo.py:39-40` only forwards `providers`). | `src/services/model_manager.py:97`, `insightface/model_zoo/model_zoo.py:39` | Build the two sessions directly with `intra_op=1`, `inter_op=1`, spinning off. Set `cv2.setNumThreads(1)`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. |
| 2 | **Startup loads 5 models, keeps 2.** `FaceAnalysis.__init__` creates an ORT session for every `.onnx` in the dir (incl. 137 MB `1k3d68.onnx`), then deletes 3. | `face_analysis.py:27-40` | Load only `det_10g.onnx` and `w600k_r50.onnx`. |
| 3 | **Image decode runs on the event loop.** `cv2.imdecode` of a 12 MP JPEG takes 80–150 ms and blocks all other requests, health checks, and downloads. | `src/api/routes_async.py:148` | Move decode + validation + inference into one thread-pool job. |
| 4 | **No back-pressure.** Under a burst, requests queue without limit in the executor; `asyncio.wait_for` cannot cancel a running thread, so each waits up to 30 s then fails. | `src/services/inference_executor.py:86-96` | Bounded slots (`pool + max_queue`). Reject extra requests fast with **503 + Retry-After**. |
| 5 | **Recognition runs one ONNX call per face.** `w600k_r50.onnx` has a dynamic batch dim (`['None',3,112,112]`) and `ArcFaceONNX.get_feat` already accepts a list. | `face_analysis.py:52-63`, `arcface_onnx.py:77-85` | Detect, crop all faces, one batched recognition call. |
| 6 | **Cold start downloads 280 MB zip** (no volume in Coolify). | `Dockerfile:56`, `storage.py:9-25` | Download the 2 needed files at image build time. |
| 7 | **Large JPEGs decoded at full size.** Detection resizes to 640 anyway (`retinaface.py:220`). A 4000×3000 image costs ~36 MB and ~100 ms decode per worker. | `src/services/image.py:141` | For JPEGs with longest side ≥ 4×`det_size` (the existing warning threshold), decode with `IMREAD_REDUCED_COLOR_2` and scale bbox/kps back ×2. |
| 8 | **Response payload is 2× larger than needed.** float32 → Python float prints 17 digits. 10 faces = 105 KB. Serialization also does validate + `model_dump` + `json.dumps` (1.4 ms). | `src/services/image.py:213` | Round embeddings to 6 decimals in float64 (measured: 10.6 KB → 5.3 KB per face). Return `model_dump_json()` directly. |
| 9 | **Uploads > 500 KB spool to disk.** Werkzeug `default_stream_factory` uses `SpooledTemporaryFile(max_size=500 KB)`; Quart’s `FormDataParser` accepts a `stream_factory`. | `werkzeug/formparser.py:59-62`, `quart/formparser.py:52` | Custom `Request` subclass with an in-memory stream factory. Low priority; URL input mode makes this path less used. |
| 10 | **First request pays ORT warm-up** (arena allocation, ~0.5–2 s). | — | Run one dummy detection and one dummy recognition after load. |

Not changed (documented tradeoffs only): `det_size=640` (lowering to 480 gives ~40 % faster detection but misses small faces), New Relic agent overhead (a few % CPU, keep).

## Implementation

### Step 1 — Config (`src/config/settings.py`)

Add fields:

- `det_model_file: str = "det_10g.onnx"`, `rec_model_file: str = "w600k_r50.onnx"`
- `ort_intra_op_threads: int = 1` (ge=1, le=32)
- `inference_max_queue: int = 16` (ge=0)
- `image_url_allowed_hosts: str = ""` — comma-separated host patterns (`*.r2.cloudflarestorage.com`). Empty = URL input disabled.
- `download_timeout: float = 10.0`, `download_max_concurrency: int = 16`
- `embedding_decimals: int = 6`

Document them in `AGENTS.md` (Configuration section) and `README.md`. Also fix the AGENTS.md claim that the model loads lazily; it loads at startup in `create_async_app`.

### Step 2 — Model loading and batched inference (`src/services/model_manager.py`)

Replace `insightface.app.FaceAnalysis` with a small internal engine:

- `_make_session(path, settings)`: `ort.SessionOptions()` with `intra_op_num_threads=settings.ort_intra_op_threads`, `inter_op_num_threads=1`, `execution_mode=ORT_SEQUENTIAL`, `add_session_config_entry("session.intra_op.allow_spinning", "0")`, `providers=[settings.execution_provider.value]`.
- `_FaceEngine(det, rec)` with `get(img) -> list[Face]`:
  1. `bboxes, kpss = det.detect(img, max_num=0, metric="default")` (same as `FaceAnalysis.get`)
  2. Build `insightface.app.common.Face` objects.
  3. `crops = [face_align.norm_crop(img, landmark=f.kps, image_size=112) for f in faces]`
  4. `feats = rec.get_feat(crops)` — **one** ONNX call, shape `(N, 512)`; assign `face.embedding = feats[i]`.
  5. `warmup()`: `det.detect(zeros 640×640)` and `rec.get_feat([zeros 112×112])`.
- `ModelManager.load()`:
  - `cv2.setNumThreads(1)`.
  - `model_dir = insightface.utils.ensure_available("models", model_name, root=model_root)` (keeps the download fallback for local dev).
  - Build `RetinaFace(model_file=..., session=...)`, `.prepare(0, input_size=(d, d), det_thresh=...)`; `ArcFaceONNX(model_file=..., session=...)`, `.prepare(0)`.
  - `self.model = _FaceEngine(det, rec)`; call `warmup()`. Keep `self.model.get(img)` as the public shape so `get_faces()` and tests (`manager.model.get.return_value`) stay unchanged.
- Imports of `insightface` and `onnxruntime` stay inside `load()` (test mocking).

### Step 3 — Back-pressure (`src/services/inference_executor.py`, `src/exceptions/errors.py`)

- New `ServiceOverloadedError(APIError)` → 503, `error_code="OVERLOADED"`.
- `InferenceExecutor.__init__(max_workers, timeout, max_queue=16)`: `self._slots = asyncio.Semaphore(max_workers + max_queue)`.
- `run()`: `if self._slots.locked(): raise ServiceOverloadedError(...)`; then `async with self._slots:` around the existing `wait_for`.
- In `src/app_async.py` error handler for `APIError`: add header `Retry-After: 1` when status is 503. Wire `max_queue=settings.inference_max_queue`.

### Step 4 — One thread-pool job per request (`src/services/pipeline.py`, new)

`process_image(file_bytes, mm, settings) -> PipelineResult` (dataclass: `faces`, `width`, `height`, `decode_ms`, `detect_ms`). Runs entirely in a worker thread:

1. `verify_magic_bytes` (reuse from `image.py`).
2. Read dimensions from the header only (`PIL.Image.open(BytesIO(b)).size` — lazy, no pixel decode; Pillow is already an `insightface` dependency, pin it in `requirements.txt`).
3. If JPEG and `max(w, h) >= 4 * settings.max_image_dimension`: `cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)`, `scale = 2`. Else `IMREAD_COLOR`, `scale = 1`. Keep the existing 3-channel check.
4. `validate_image_dimensions` (reuse).
5. `faces = mm.get_faces(image)`; if `scale != 1`: `face.bbox *= scale`, `face.kps *= scale` (return coordinates in original pixels).
6. Return faces + timings. `extract_face_data` runs here too (also CPU).

Keep `decode_image()` unchanged for the Flask app and existing tests.

### Step 5 — URL input (`src/services/image_fetch.py`, new; `src/app_async.py`)

- Add `httpx==0.28.1` to `requirements.txt` (`h11`, `anyio`, `certifi` already present).
- `create_async_app`: `@app.before_serving` creates one `httpx.AsyncClient(timeout=settings.download_timeout, follow_redirects=False, limits=Limits(max_connections=download_max_concurrency))`; `@app.after_serving` closes it. Store in `app.config["http_client"]`. Create `asyncio.Semaphore(download_max_concurrency)`.
- `fetch_image(url, client, semaphore, settings) -> bytes`:
  - Parse with `urllib.parse.urlsplit`; require scheme `https`; host must match `image_url_allowed_hosts` via `fnmatch`. Empty allowlist → `RequestValidationError("image_url input is disabled")`. Reject IP-literal hosts.
  - Stream the response; abort if `Content-Length` or the byte count exceeds `max_content_length`.
  - Map network/HTTP errors to new `ImageDownloadError(APIError)` → 502, `error_code="IMAGE_DOWNLOAD_FAILED"`.

### Step 6 — Route (`src/api/routes_async.py`)

`/represent`:

1. Readiness check (unchanged).
2. Input: if `request.content_type` is JSON (or form field `image_url` present) → `fetch_image(...)`; else existing `request.files["image_file"]` path with `validate_image_file` (reuse). Exactly one of the two is required.
3. `result = await executor.run(process_image, file_bytes, mm, settings)` wrapped in **one** `trace("inference", "Custom/Inference")`.
4. NR attributes/metrics: reuse the current names; `detection_time_ms` from `result.detect_ms`, add `decode_time_ms` and `input_mode` (`file`|`url`).
5. Response: `Response(RepresentResponse(...).model_dump_json(), content_type="application/json")`.

`extract_face_data` (`src/services/image.py`): `embedding=np.round(face.embedding.astype(np.float64), decimals).tolist()`.

### Step 7 — In-memory upload buffer (`src/app_async.py`) — low priority

`class InMemoryRequest(Request)` overriding `make_form_data_parser()` to pass `stream_factory=lambda **_: io.BytesIO()`; set `app.request_class = InMemoryRequest`. `MAX_CONTENT_LENGTH` (16 MB) already bounds memory.

### Step 8 — Docker and start script

`Dockerfile`:

- Builder stage: `curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` → `unzip -j ... det_10g.onnx w600k_r50.onnx -d /models` (add `unzip` to builder apt list). Runtime: `COPY --from=builder --chown=appuser /models /app/insightface/models/buffalo_l/`.
- `ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

`bin/start`: print `INFERENCE_POOL_SIZE`, `ORT_INTRA_OP_THREADS`, `INFERENCE_MAX_QUEUE` instead of the hard-coded "4 threads" line. Keep `uvicorn` flags (uvloop/httptools auto-enabled by `uvicorn[standard]`).

### Step 9 — Tests

- `tests/conftest.py`: add `sys.modules["onnxruntime"] = MagicMock()` next to the insightface mocks so `load()` builds mock sessions.
- `tests/test_concurrency.py:46,72`: patch target changes from `insightface.app.FaceAnalysis` to `src.services.model_manager._make_session` (count calls / raise).
- New tests: executor returns 503 when slots are full; `process_image` scales bbox/kps ×2 for a large JPEG (generate with `cv2.imencode`); `fetch_image` rejects `http://`, IP hosts, non-allowlisted hosts, oversized bodies (mock `httpx.AsyncClient`); `extract_face_data` rounds to 6 decimals; route accepts `image_url` and rejects both/none inputs.
- Update existing assertions that compare `embedding == face.embedding.tolist()`.

### Step 10 — Benchmark script (`bin/bench.py`)

Small `httpx` script: N concurrent POSTs of one sample image, prints p50/p95 latency and req/s. Used to confirm the thread split on the VPS (`INFERENCE_POOL_SIZE=4 ORT_INTRA_OP_THREADS=1` vs `2×2`).

## Verification

1. The local `venv` is broken (symlink to a removed asdf Python). Recreate: `python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt`.
2. `./venv/bin/python -m pytest tests/ -v` — all green, including new tests.
3. Real model, local: `./bin/start`, then
   - `curl -F image_file=@photo.jpg localhost:5001/represent` → same faces as before; bbox in original pixel coordinates; embedding values have ≤ 6 decimals.
   - `curl -H 'content-type: application/json' -d '{"image_url":"https://<allowed-host>/photo.jpg"}' localhost:5001/represent` → 200; with a non-allowlisted host → 400.
   - Startup log shows load time and that only 2 sessions are created (no `1k3d68` line).
4. Burst: `./venv/bin/python bin/bench.py --concurrency 40 --requests 200` → some 503s with `Retry-After`, no 504s, p95 bounded; `top` shows ~400 % CPU total and no more than ~6 busy threads.
5. Docker: `docker build -t insightface-api .`; `docker run -p 5001:5001 insightface-api` with no volume → `/health` healthy in < 15 s with no download log line.
6. Compare embeddings old vs new for the same image: cosine similarity ≥ 0.9999 (batched recognition + rounding must not change results materially).
