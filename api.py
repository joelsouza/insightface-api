import flask
from flask import Flask, request, jsonify
import insightface
import numpy as np
import cv2
import requests
import os
import logging
import time

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application Configuration ---
app = Flask(__name__)

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
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    end_time = time.time()
    logging.info(
        f"InsightFace model loaded in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logging.error(f"Error loading InsightFace model: {e}")
    # If the model doesn't load, the API cannot function. Could exit or let routes return errors.
    face_app = None  # Indicates that the model is not ready

# --- API Endpoint ---


@app.route('/extract_embeddings', methods=['POST'])
def extract_embeddings():
    """
    Endpoint to extract facial embeddings from an image provided by URL.
    Receives: JSON with {"image_url": "IMAGE_URL"}
    Returns: JSON with {"embeddings": [[embedding1], [embedding2], ...]} or {"error": "message"}
    """
    if face_app is None:
        logging.error(
            "Request received, but the InsightFace model was not loaded.")
        return jsonify({"error": "InsightFace model not initialized correctly."}), 500

    # 1. Get URL from JSON request
    if not request.is_json:
        logging.warning("Received request is not JSON.")
        return jsonify({"error": "Request must be JSON type."}), 400

    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        logging.warning("Field 'image_url' missing in request.")
        return jsonify({"error": "'image_url' is required in the request body."}), 400

    logging.info(f"Processing image from URL: {image_url}")

    try:
        # 2. Download the image from URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(
            image_url, headers=headers, timeout=15, stream=True)  # 15s timeout
        response.raise_for_status()  # Raises error for bad HTTP status (4xx, 5xx)

        # Read the image content
        image_data = response.content

        # 3. Decode the image using OpenCV
        # Converts downloaded bytes to a NumPy array
        img_array = np.frombuffer(image_data, np.uint8)
        # Decodes the NumPy array into an OpenCV image (BGR format by default)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logging.error(f"Failed to decode image from URL: {image_url}")
            return jsonify({"error": "Could not decode the image. Verify if the URL points to a valid image."}), 400

        # 4. Detect faces and extract embeddings using InsightFace
        logging.info("Running facial analysis...")
        start_time_analysis = time.time()
        # The get() method returns a list of 'Face' objects
        faces = face_app.get(img)
        end_time_analysis = time.time()
        logging.info(
            f"Facial analysis completed in {end_time_analysis - start_time_analysis:.2f} seconds. Faces found: {len(faces)}")

        # 5. Prepare the response
        embeddings = []
        if faces:
            for face in faces:
                # The 'embedding' attribute is a NumPy array
                embedding_np = face.embedding
                # Normalize the embedding (optional, but good practice for cosine comparisons)
                # embedding_norm = embedding_np / np.linalg.norm(embedding_np)
                # Convert to list of floats to be serializable in JSON
                # use embedding_norm if normalized
                embeddings.append(embedding_np.tolist())

        logging.info(f"Embeddings extracted: {len(embeddings)}")
        return jsonify({"embeddings": embeddings})

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading image from {image_url}: {e}")
        return jsonify({"error": f"Error downloading image from URL: {e}"}), 400
    except cv2.error as e:
        logging.error(
            f"OpenCV error processing image from {image_url}: {e}")
        return jsonify({"error": f"Error processing image with OpenCV: {e}"}), 500
    except Exception as e:
        # Captures any other unexpected error during the process
        # Uses logging.exception to include traceback
        logging.exception(f"Unexpected error processing {image_url}: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500


# --- Application Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the API accessible on your local network
    # debug=True automatically reloads on changes, but DO NOT use in production
    port = int(os.environ.get('PORT', 5000))  # Default port 5000
    # Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)
