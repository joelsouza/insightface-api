from flask import Flask, request, jsonify
import insightface
import cv2
import os
import logging
import time
import gc
import time
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024

providers = ['CPUExecutionProvider']
max_image_width = 640
max_image_height = 640

try:
    logging.info("Loading InsightFace model 'buffalo_l'...")
    start_time = time.time()
    face_app = insightface.app.FaceAnalysis(name='buffalo_l',
                                            root='./insightface',
                                            allowed_modules=['detection', 'recognition'],
                                            providers=providers)
    face_app.prepare(ctx_id=0, det_size=(max_image_width, max_image_height), det_thresh=0.5)
    end_time = time.time()
    logging.info(
        f"InsightFace model loaded in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logging.error(f"Error loading InsightFace model: {e}")
    face_app = None

@app.route('/represent', methods=['POST'])
def represent():
    if face_app is None:
        logging.error("InsightFace model was not loaded.")
        return jsonify({"error": "InsightFace model not initialized correctly."}), 500

    image_file = request.files.get('image_file')

    if not image_file:
        logging.warning("No image file provided.")
        return jsonify({"error": "No image file provided."}), 400

    img = None  # Initialize img to ensure cleanup
    file_bytes = None
    nparr = None
    faces = None
    embeddings = []

    try:
        file_bytes = image_file.read()

        # Decode image directly from memory
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        logging.info("Running facial analysis...")
        start_time_analysis = time.time()
        faces = face_app.get(img)
        end_time_analysis = time.time()
        logging.info(
            f"Facial analysis completed in {end_time_analysis - start_time_analysis:.2f} seconds. Faces found: {len(faces)}")

        for face in faces:
            embeddings.append({
                "embedding": face.embedding.tolist(),
                "bbox": face.bbox.astype(int).tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
                "gender": int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None,
                "age": int(face.age) if hasattr(face, 'age') and face.age is not None else None
            })

        logging.info(f"Embeddings extracted: {len(embeddings)}")

        return jsonify({"embeddings": embeddings})

    finally:
        if img is not None:
            del img
        if file_bytes is not None:
            del file_bytes
        if nparr is not None:
            del nparr
        if faces is not None:
            del faces

        embeddings.clear()

        gc.collect()

@app.route('/up', methods=['GET'])
def up():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
