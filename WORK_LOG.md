# Work Log - NewRelic Instrumentation

## Date: 2025-12-09

### Task: Instrument InsightFace API with NewRelic

#### Changes Made

1. **requirements.txt** - Added dependencies:
   - `newrelic==10.4.0` - NewRelic Python agent
   - `psutil==5.9.8` - System metrics collection

2. **newrelic.ini** - Created NewRelic configuration file with:
   - Application name: "InsightFace API"
   - Transaction tracing enabled
   - Distributed tracing enabled
   - Custom events and metrics enabled
   - Environment-specific settings (development, staging, production)

3. **instrumentation.py** - Created comprehensive instrumentation module:
   - `trace_segment()` - Context manager for tracing code segments
   - `trace_function()` - Decorator for function tracing
   - `PipelineTracer` - Class to trace multi-step pipelines with step-by-step metrics
   - `RequestMetrics` - Class to capture per-request metrics
   - Helper functions for face detection, embedding, and comparison metrics
   - Memory tracking via psutil

4. **api.py** - Instrumented with:
   - NewRelic initialization at startup
   - Pipeline tracing for the `/represent` endpoint with steps:
     - `read_file` - File reading time
     - `decode_image` - Image decoding time
     - `resize_image` - Image resizing time
     - `face_detection` - Face detection time (main bottleneck)
     - `extract_embeddings` - Embedding extraction time
   - Model loading instrumentation
   - Custom attributes for image dimensions, faces detected, file size
   - `/metrics` endpoint for runtime inspection

5. **Dockerfile** - Updated with:
   - NewRelic environment variables
   - Config file generation step

6. **start.sh** - Updated to:
   - Use `newrelic-admin run-program` wrapper when license key is set
   - Fall back to standard gunicorn when NewRelic is not configured

7. **fly.toml** - Added NewRelic environment configuration

#### Custom Metrics Available in NewRelic

| Metric Path | Description |
|------------|-------------|
| `Custom/Pipeline/face_representation/read_file/duration_ms` | Time to read uploaded file |
| `Custom/Pipeline/face_representation/decode_image/duration_ms` | Time to decode image |
| `Custom/Pipeline/face_representation/resize_image/duration_ms` | Time to resize image |
| `Custom/Pipeline/face_representation/face_detection/duration_ms` | Time for face detection |
| `Custom/Pipeline/face_representation/extract_embeddings/duration_ms` | Time to extract embeddings |
| `Custom/FaceDetection/detection_time_ms` | Total detection time |
| `Custom/FaceDetection/faces_detected` | Number of faces detected |
| `Custom/FaceDetection/pixels_per_ms` | Processing throughput |
| `Custom/Request/represent/duration_ms` | Total request duration |
| `Custom/Request/represent/memory_delta_mb` | Memory usage change |

#### Custom Events

- `PipelineExecution` - Summary of each pipeline run
- `APIRequest` - Per-request metrics and attributes
- `FaceDetection` - Detection-specific metrics
- `FaceEmbedding` - Embedding extraction metrics

#### Deployment Steps

1. Set NewRelic license key as a Fly.io secret:
   ```bash
   fly secrets set NEW_RELIC_LICENSE_KEY=your_license_key_here
   ```

2. Deploy the application:
   ```bash
   fly deploy
   ```

3. Access NewRelic dashboard to view:
   - APM > Transactions for endpoint performance
   - APM > Distributed Tracing for request traces
   - Custom Dashboards for pipeline step analysis
   - Insights > Data Explorer for custom events
