import os
import torch
import open3d as o3d
from flask import Flask, request, jsonify, send_from_directory
import argparse
import uuid
from hy3dworld import LayerDecomposition, WorldComposer, process_file

# --- Model Loading ---
# We'll load the models based on the 'resolution' parameter in the request.
# This avoids loading both high and low-res models into memory at the same time.
models = {
    "low": None,
    "high": None
}

def get_model(resolution="low", seed=42):
    if models[resolution] is None:
        if resolution == "low":
            from demo_scenegen_lowRes import HYworldDemo
            models[resolution] = HYworldDemo(seed=seed)
        else:
            # Assuming you have a 'demo_scenegen.py' for high-res
            from demo_scenegen import HYworldDemo
            models[resolution] = HYworldDemo(seed=seed)
    return models[resolution]

# --- Flask App ---
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # --- Get parameters from the request ---
    image = request.files['image']
    labels_fg1 = request.form.getlist('labels_fg1[]')
    labels_fg2 = request.form.getlist('labels_fg2[]')
    resolution = request.form.get('resolution', 'low')
    classes = request.form.get('classes', 'outdoor')
    seed = int(request.form.get('seed', 42))
    use_sr = request.form.get('use_sr', 'False').lower() in ('true', '1', 't')
    export_drc = request.form.get('export_drc', 'False').lower() in ('true', '1', 't')

    # --- Prepare for generation ---
    request_id = str(uuid.uuid4())
    output_dir = os.path.join('output', request_id)
    os.makedirs(output_dir, exist_ok=True)
    
    image_path = os.path.join(output_dir, image.filename)
    image.save(image_path)

    try:
        # --- Run the model ---
        model = get_model(resolution, seed)
        output_files = model.run(
            image_path=image_path,
            labels_fg1=labels_fg1,
            labels_fg2=labels_fg2,
            classes=classes,
            output_dir=output_dir,
            export_drc=export_drc,
            use_sr=use_sr
        )
        
        # --- Prepare the response ---
        file_urls = [f"/output/{request_id}/{os.path.basename(f)}" for f in output_files]
        return jsonify({'message': 'Generation successful', 'files': file_urls})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<path:path>')
def send_output(path):
    return send_from_directory('output', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HunyuanWorld API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)
