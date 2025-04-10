from flask import Flask, request, jsonify
import insightface
import numpy as np
import cv2
import requests
import os
import logging
import time
import gc

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application Configuration ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit.
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024  # 16MB max-limit.

# --- InsightFace Model Loading ---
# Loads the model once when the application starts to avoid reloading with each request.
# 'buffalo_l' is a good and comprehensive set of models (detection, alignment, recognition).
# 'providers' defines the execution preference order (CUDA for GPU, then CPU).
# Adjust 'providers' according to your installation (remove 'CUDAExecutionProvider' if using CPU only).
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
providers = ['CPUExecutionProvider']

try:
    logging.info("Loading InsightFace model 'buffalo_l'...")
    start_time = time.time()
    # Increasing det_thresh can reduce false positives, decreasing can find more difficult faces.
    # det_score is the correct parameter name in newer versions for the detection score threshold.
    face_app = insightface.app.FaceAnalysis(name='buffalo_l',
                                            # Ensures we have detection and recognition
                                            allowed_modules=[
                                                'detection', 'recognition'],
                                            providers=providers)
    # ctx_id=0 usually refers to the first GPU or CPU. det_size is important for detection.
    face_app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.5)
    end_time = time.time()
    logging.info(
        f"InsightFace model loaded in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logging.error(f"Error loading InsightFace model: {e}")
    # If the model doesn't load, the API cannot function. Could exit or let routes return errors.
    face_app = None  # Indicates that the model is not ready

# --- API Endpoint ---


@app.route('/represent', methods=['POST'])
def represent():
    """
    Endpoint to extract facial embeddings from an image.
    Receives: JSON with {"image_url": "IMAGE_URL"} or {"image_data": "BASE64_ENCODED_IMAGE"}
    Returns: JSON with {"embeddings": [[embedding1], [embedding2], ...]} or {"error": "message"}
    """
    if face_app is None:
        logging.error(
            "Request received, but the InsightFace model was not loaded.")
        return jsonify({"error": "InsightFace model not initialized correctly."}), 500

    # 1. Get image_data from POST body
    image_data = request.form.get('image_data')

    if not image_data:
        logging.warning("No image data provided.")
        return jsonify({"error": "No image data provided."}), 400

    img = None
    faces = None
    img_array = None
    decoded_data = None
    embeddings = []

    try:
        # Process image data directly from request
        logging.info("Processing image from provided image_data")
        import base64
        # Decode base64 image data
        try:
            # Handle potential padding issues
            image_data = image_data.replace(' ', '+')
            # Remove potential data URL prefix
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]

            # Decode base64 to bytes
            decoded_data = base64.b64decode(image_data)

            # Convert to numpy array and decode with OpenCV
            img_array = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logging.error("Failed to decode image from provided image_data")
                return jsonify({"error": "Could not decode the image data. Verify the format."}), 400
        except Exception as e:
            logging.error(f"Error decoding image_data: {e}")
            return jsonify({"error": f"Error decoding image data: {e}"}), 400


        # 4. Detect faces and extract embeddings using InsightFace
        logging.info("Running facial analysis...")
        start_time_analysis = time.time()
        faces = face_app.get(img)
        end_time_analysis = time.time()
        logging.info(
            f"Facial analysis completed in {end_time_analysis - start_time_analysis:.2f} seconds. Faces found: {len(faces)}")

        # 5. Prepare the response
        if faces:
            for face in faces:
                embeddings.append({
                    "embedding": face.embedding.tolist(),
                    "bbox": face.bbox.tolist(),
                    "kps": face.kps.tolist(),
                    "det_score": float(face.det_score),
                    "face_image_bytes": image_data
                })

        logging.info(f"Embeddings extracted: {len(embeddings)}")

        # Liberar memória de forma explícita
        decoded_data = None
        img_array = None
        img = None

        return jsonify({"embeddings": embeddings})

    except requests.exceptions.RequestException as e:
        logging.error(f"OpenCV error processing image: {e}")
        return jsonify({"error": f"Error processing image with OpenCV: {e}"}), 500
    except Exception as e:
        # Captures any other unexpected error during the process
        logging.exception(f"Unexpected error processing: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500
    finally:
        # Liberar memória de forma explícita
        decoded_data = None
        img_array = None
        img = None
        faces = None
        embeddings = None
        gc.collect()


# --- Application Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the API accessible on your local network
    # debug=True automatically reloads on changes, but DO NOT use in production
    port = int(os.environ.get('PORT', 5001))  # Default port 5001
    # Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)
