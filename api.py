from flask import Flask, request, jsonify
import insightface
import numpy as np
import cv2
import os
import logging
import time
import gc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024

providers = ['CPUExecutionProvider']
max_image_width = 1280
max_image_height = 1280

try:
    logging.info("Loading InsightFace model 'buffalo_l'...")
    start_time = time.time()
    face_app = insightface.app.FaceAnalysis(name='buffalo_l',
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

    image_path = None

    try:
        file_bytes = image_file.read()

        # Create tmp directory if it doesn't exist
        tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        image_path = os.path.join(tmp_dir, f"{time.time()}.jpg")
        with open(image_path, 'wb') as f:
            f.write(file_bytes)

        img = cv2.imread(image_path)

        logging.info("Running facial analysis...")
        start_time_analysis = time.time()
        faces = face_app.get(img)
        end_time_analysis = time.time()
        logging.info(
            f"Facial analysis completed in {end_time_analysis - start_time_analysis:.2f} seconds. Faces found: {len(faces)}")

        embeddings = []
        for face in faces:
            embeddings.append({
                "embedding": face.embedding.tolist(),
                "bbox": face.bbox.astype(int).tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
                "gender": int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None,
                "age": int(face.age) if hasattr(face, 'age') and face.age is not None else None,
            })

        logging.info(f"Embeddings extracted: {len(embeddings)}")

        return jsonify({"embeddings": embeddings})

    finally:
        # Clean up the temporary file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logging.error(f"Error removing temporary file: {e}")

        # Release memory
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
