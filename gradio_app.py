import gradio as gr
import requests
import json
from PIL import Image
import io
import os

# --- Configuration ---
# You'll need to replace this with your RunPod API endpoint
RUNPOD_API_URL = "http://205.196.17.28:8000/generate" 

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# HunyuanWorld-1.0 Gradio UI")

        with gr.Row():
            test_button = gr.Button("Test Connection")
            test_status = gr.Textbox(label="Connection Status", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Text-to-World"):
                with gr.Row():
                    with gr.Column():
                        # Inputs
                        prompt = gr.Textbox(label="Prompt")
                        negative_prompt = gr.Textbox(label="Negative Prompt")
                        labels_fg1 = gr.Textbox(label="Foreground Labels (Layer 1)", placeholder="e.g., person, car")
                        labels_fg2 = gr.Textbox(label="Foreground Labels (Layer 2)", placeholder="e.g., tree, building")
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            resolution = gr.Radio(["low", "high"], label="Resolution", value="low")
                            classes = gr.Dropdown(["outdoor", "indoor"], label="Classes", value="outdoor")
                            seed = gr.Slider(0, 10000, value=42, label="Seed", step=1)
                            use_sr = gr.Checkbox(label="Use Super Resolution", value=False)
                            export_drc = gr.Checkbox(label="Export Draco (.drc)", value=False)

                        # Button
                        generate_button = gr.Button("Generate")

                    with gr.Column():
                        # Outputs
                        output_image = gr.Image(label="Generated Panorama")
                        output_files = gr.File(label="Download 3D Model")
                        status = gr.Textbox(label="Status", interactive=False)

                # Click action
                generate_button.click(
                    fn=text_to_world,
                    inputs=[prompt, negative_prompt, labels_fg1, labels_fg2, resolution, classes, seed, use_sr, export_drc],
                    outputs=[output_image, output_files, status]
                )

            with gr.TabItem("Image-to-World"):
                with gr.Row():
                    with gr.Column():
                        # Inputs
                        input_image = gr.Image(type="pil", label="Input Image")
                        img_prompt = gr.Textbox(label="Prompt")
                        img_negative_prompt = gr.Textbox(label="Negative Prompt")
                        img_labels_fg1 = gr.Textbox(label="Foreground Labels (Layer 1)", placeholder="e.g., person, car")
                        img_labels_fg2 = gr.Textbox(label="Foreground Labels (Layer 2)", placeholder="e.g., tree, building")

                        with gr.Accordion("Advanced Settings", open=False):
                            img_resolution = gr.Radio(["low", "high"], label="Resolution", value="low")
                            img_classes = gr.Dropdown(["outdoor", "indoor"], label="Classes", value="outdoor")
                            img_seed = gr.Slider(0, 10000, value=42, label="Seed", step=1)
                            img_use_sr = gr.Checkbox(label="Use Super Resolution", value=False)
                            img_export_drc = gr.Checkbox(label="Export Draco (.drc)", value=False)
                        
                        # Button
                        img_generate_button = gr.Button("Generate")

                    with gr.Column():
                        # Outputs
                        img_output_image = gr.Image(label="Generated Panorama")
                        img_output_files = gr.File(label="Download 3D Model")
                        img_status = gr.Textbox(label="Status", interactive=False)

                # Click action
                img_generate_button.click(
                    fn=image_to_world,
                    inputs=[input_image, img_prompt, img_negative_prompt, img_labels_fg1, img_labels_fg2, img_resolution, img_classes, img_seed, img_use_sr, img_export_drc],
                    outputs=[img_output_image, img_output_files, img_status]
                )
        
        test_button.click(
            fn=test_connection,
            outputs=[test_status]
        )

    return demo

def test_connection():
    try:
        response = requests.get(RUNPOD_API_URL.replace("/generate", "/test"), timeout=10)
        response.raise_for_status()
        return response.json().get('message', 'Connection successful!')
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the RunPod API: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

def text_to_world(prompt, negative_prompt, labels_fg1, labels_fg2, resolution, classes, seed, use_sr, export_drc):
    # This function will be more complex, for now, it's a placeholder
    return None, None, "Text-to-World generation is not yet implemented."


def image_to_world(image, prompt, negative_prompt, labels_fg1, labels_fg2, resolution, classes, seed, use_sr, export_drc):
    if image is None:
        return None, None, "Please upload an image."

    # Convert PIL Image to bytes
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()

    files = {'image': ('image.png', byte_arr, 'image/png')}
    data = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'labels_fg1': labels_fg1.split(','),
        'labels_fg2': labels_fg2.split(','),
        'resolution': resolution,
        'classes': classes,
        'seed': seed,
        'use_sr': use_sr,
        'export_drc': export_drc
    }

    try:
        response = requests.post(RUNPOD_API_URL, files=files, data=data, timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        
        # For now, we'll just show the files. We can later add logic to download and display them.
        files = result.get('files', [])
        
        # A real implementation would download the panorama and display it.
        # For now, we'll just return the file list.
        return None, files, f"Generation successful. Files: {', '.join(files)}"

    except requests.exceptions.RequestException as e:
        return None, None, f"Error connecting to the RunPod API: {e}"
    except Exception as e:
        return None, None, f"An error occurred: {e}"


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
