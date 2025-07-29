import os
import torch
import open3d as o3d
from flask import Flask, request, jsonify, send_from_directory
import argparse
import uuid
from hy3dworld import LayerDecomposition, WorldComposer, process_file
from demo_panogen_lowRes import Text2PanoramaDemo

# --- Model Loading ---
# We'll load the models based on the 'resolution' parameter in the request.
# This avoids loading both high and low-res models into memory at the same time.
models = {
    "low": None,
    "high": None,
    "text2pano": None
}

def get_model(model_type="low", resolution="low", seed=42):
    if model_type == "text2pano":
        if models["text2pano"] is None:
            # Assuming low-res for now, can be parameterized later
            width, height = (1024, 512) if resolution == "low" else (2048, 1024)
            models["text2pano"] = Text2PanoramaDemo(width=width, height=height)
        return models["text2pano"]
    
    # For scenegen models
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
    # --- Get parameters from the request ---
    is_text_to_world = 'image' not in request.files

    if is_text_to_world:
        prompt = request.form.get('prompt')
        negative_prompt = request.form.get('negative_prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided for text-to-world generation'}), 400
    else:
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
    
    image_path = None
    if is_text_to_world:
        try:
            # --- Generate Panorama from Text ---
            pano_model = get_model(model_type="text2pano", resolution=resolution, seed=seed)
            # Save panorama directly to the request's output directory
            pano_image = pano_model.run(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                output_path=output_dir
            )
            image_path = os.path.join(output_dir, 'panorama.png')
        except Exception as e:
            return jsonify({'error': f'Panorama generation failed: {str(e)}'}), 500
    else:
        # --- Save uploaded image ---
        image_path = os.path.join(output_dir, image.filename)
        image.save(image_path)

    try:
        # --- Run the Scene Generation model ---
        model = get_model(model_type=resolution, resolution=resolution, seed=seed)
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
        # Also include the generated panorama if it was a text-to-world request
        if is_text_to_world:
            output_files.append(image_path)
            
        file_urls = [f"/output/{request_id}/{os.path.basename(f)}" for f in output_files]
        
        # Special handling to return the panorama image for display in Gradio
        panorama_url = None
        if is_text_to_world:
            panorama_url = f"/output/{request_id}/panorama.png"

        return jsonify({
            'message': 'Generation successful', 
            'files': file_urls,
            'panorama_image_url': panorama_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Connection to RunPod API successful!'})

@app.route('/output/<path:path>')
def send_output(path):
    return send_from_directory('output', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HunyuanWorld API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)
