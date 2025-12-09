import os

# Initialize NewRelic before any other imports
from instrumentation import initialize_newrelic
nr_environment = os.environ.get('NEW_RELIC_ENVIRONMENT', 'production')
initialize_newrelic('newrelic.ini', nr_environment)

from flask import Flask, request, jsonify
import insightface
import cv2
import logging
import time
import gc
import numpy as np

from instrumentation import (
    PipelineTracer,
    RequestMetrics,
    trace_segment,
    record_face_detection_metrics,
    record_embedding_metrics,
    add_custom_attribute,
    get_memory_usage_mb,
    record_custom_metric
)

try:
    import newrelic.agent as nr_agent
    NEWRELIC_AVAILABLE = True
except ImportError:
    NEWRELIC_AVAILABLE = False
    nr_agent = None

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024

providers = ['CPUExecutionProvider']
max_image_width = 640
max_image_height = 640

face_app = None


def load_model():
    """Load or reload the InsightFace model."""
    global face_app

    with trace_segment("load_model", "ModelManagement"):
        if face_app is not None:
            logging.info("Cleaning up existing model resources...")
            with trace_segment("cleanup_existing_model", "ModelManagement"):
                del face_app
                gc.collect()

        try:
            logging.info("Loading InsightFace model 'buffalo_l'...")
            start_time = time.time()
            start_memory = get_memory_usage_mb()

            with trace_segment("model_initialization", "ModelManagement"):
                face_app = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    root='./insightface',
                    allowed_modules=['detection', 'recognition'],
                    providers=providers
                )

            with trace_segment("model_prepare", "ModelManagement"):
                face_app.prepare(
                    ctx_id=0,
                    det_size=(max_image_width, max_image_height),
                    det_thresh=0.5
                )

            end_time = time.time()
            end_memory = get_memory_usage_mb()
            load_duration = end_time - start_time
            memory_used = end_memory - start_memory

            logging.info(f"InsightFace model loaded in {load_duration:.2f} seconds.")
            logging.info(f"Model memory usage: {memory_used:.2f} MB")

            record_custom_metric("ModelManagement/load_duration_seconds", load_duration)
            record_custom_metric("ModelManagement/model_memory_mb", memory_used)

        except Exception as e:
            logging.error(f"Error loading InsightFace model: {e}")
            face_app = None


load_model()


@app.route('/represent', methods=['POST'])
def represent():
    request_metrics = RequestMetrics("represent")
    tracer = PipelineTracer("face_representation")

    if face_app is None:
        logging.error("InsightFace model was not loaded.")
        request_metrics.set("error", "model_not_initialized")
        request_metrics.finish()
        return jsonify({"error": "InsightFace model not initialized correctly."}), 500

    image_file = request.files.get('image_file')

    if not image_file:
        logging.warning("No image file provided.")
        request_metrics.set("error", "no_image_file")
        request_metrics.finish()
        return jsonify({"error": "No image file provided."}), 400

    try:
        with tracer.step("read_file"):
            file_bytes = image_file.read()
            file_size_kb = len(file_bytes) / 1024
            request_metrics.set("file_size_kb", file_size_kb)
            add_custom_attribute("file_size_kb", file_size_kb)

        with tracer.step("decode_image"):
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logging.warning("Failed to decode image file.")
            request_metrics.set("error", "decode_failed")
            request_metrics.finish()
            return jsonify({"error": "Failed to decode image. Invalid or corrupted file."}), 400

        image_height, image_width = img.shape[:2]
        image_pixels = image_width * image_height
        request_metrics.set("image_width", image_width)
        request_metrics.set("image_height", image_height)
        request_metrics.set("image_pixels", image_pixels)

        add_custom_attribute("image_width", image_width)
        add_custom_attribute("image_height", image_height)
        add_custom_attribute("image_pixels", image_pixels)

        logging.info(f"Image dimensions: {image_width}x{image_height} ({image_pixels} pixels)")

        with tracer.step("resize_image"):
            original_height, original_width = img.shape[:2]
            if original_width > max_image_width or original_height > max_image_height:
                scale = min(max_image_width / original_width, max_image_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                request_metrics.set("resized", True)
                request_metrics.set("resize_scale", scale)
                add_custom_attribute("resized", True)
                add_custom_attribute("resize_scale", scale)
                logging.info(f"Image resized to {new_width}x{new_height}")
            else:
                request_metrics.set("resized", False)

        logging.info("Running facial analysis...")

        with tracer.step("face_detection"):
            start_time_analysis = time.time()
            faces = face_app.get(img)
            end_time_analysis = time.time()
            detection_time_ms = (end_time_analysis - start_time_analysis) * 1000

        faces_detected = len(faces)
        request_metrics.set("faces_detected", faces_detected)
        add_custom_attribute("faces_detected", faces_detected)

        logging.info(
            f"Facial analysis completed in {detection_time_ms:.2f} ms. Faces found: {faces_detected}"
        )

        record_face_detection_metrics(
            image_width=image_width,
            image_height=image_height,
            faces_detected=faces_detected,
            detection_time_ms=detection_time_ms
        )

        with tracer.step("extract_embeddings"):
            embedding_start = time.time()
            embeddings = []
            for face in faces:
                embeddings.append({
                    "embedding": face.embedding.tolist(),
                    "bbox": face.bbox.astype(int).tolist(),
                    "kps": face.kps.tolist(),
                    "det_score": float(face.det_score),
                    "gender": int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None,
                    "age": int(face.age) if hasattr(face, 'age') and face.age is not None else None
                })
            embedding_end = time.time()
            embedding_time_ms = (embedding_end - embedding_start) * 1000

        if faces_detected > 0:
            embedding_dim = len(embeddings[0]["embedding"]) if embeddings else 0
            record_embedding_metrics(
                face_count=faces_detected,
                embedding_dim=embedding_dim,
                embedding_time_ms=embedding_time_ms
            )

        logging.info(f"Embeddings extracted: {len(embeddings)}")

        pipeline_summary = tracer.finish()
        request_metrics.set("pipeline_total_ms", pipeline_summary["total_duration_ms"])
        request_metrics.set("pipeline_memory_delta_mb", pipeline_summary["total_memory_delta_mb"])

        for step in pipeline_summary["steps"]:
            logging.info(
                f"  Step '{step['step']}': {step['duration_ms']:.2f}ms, "
                f"memory delta: {step['memory_delta_mb']:.2f}MB"
            )

        request_metrics.finish()

        return jsonify({"embeddings": embeddings})

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        request_metrics.set("error", str(e))
        request_metrics.finish()

        if NEWRELIC_AVAILABLE and nr_agent:
            nr_agent.notice_error()

        return jsonify({"error": "Failed to process image."}), 500


@app.route('/up', methods=['GET'])
def up():
    memory_mb = get_memory_usage_mb()
    record_custom_metric("HealthCheck/memory_mb", memory_mb)
    add_custom_attribute("health_check_memory_mb", memory_mb)
    return jsonify({"status": "ok", "memory_mb": round(memory_mb, 2)})


@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint to get current application metrics."""
    memory_mb = get_memory_usage_mb()
    return jsonify({
        "memory_mb": round(memory_mb, 2),
        "model_loaded": face_app is not None,
        "max_image_dimensions": f"{max_image_width}x{max_image_height}",
        "providers": providers
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
