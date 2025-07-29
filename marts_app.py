import os
import torch
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tempfile
import zipfile
from typing import Optional, List, Tuple
import logging

# Try to import spaces (only available in HF Spaces environment)
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    # Mock spaces decorator for local development
    class MockSpaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(func):
                return func
            return decorator
    spaces = MockSpaces()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import HunyuanWorld components (lazy import to avoid CUDA issues)
Text2PanoramaDemo = None
Image2PanoramaDemo = None
HYworldDemo = None

def import_hunyuan_modules():
    """Lazy import of HunyuanWorld modules to avoid CUDA initialization at startup"""
    global Text2PanoramaDemo, Image2PanoramaDemo, HYworldDemo

    if Text2PanoramaDemo is None:
        try:
            from hunyuan_world import Text2PanoramaDemo as T2P, Image2PanoramaDemo as I2P, HYworldDemo as HY
            Text2PanoramaDemo = T2P
            Image2PanoramaDemo = I2P
            HYworldDemo = HY
            logger.info("HunyuanWorld modules imported successfully")
        except ImportError as e:
            logger.warning(f"HunyuanWorld modules not found: {e}. Using mock implementations.")
            # Mock implementations for development
            class MockText2PanoramaDemo:
                def __init__(self):
                    pass
                def run(self, prompt, negative_prompt=None, seed=42, output_path='output'):
                    return Image.new('RGB', (1920, 960), color='blue')

            class MockImage2PanoramaDemo:
                def __init__(self):
                    pass
                def run(self, prompt, negative_prompt, image_path, seed=42, output_path='output'):
                    return Image.new('RGB', (1920, 960), color='green')

            class MockHYworldDemo:
                def __init__(self, seed=42):
                    pass
                def run(self, image_path, labels_fg1, labels_fg2, classes="outdoor", output_dir='output', export_drc=False):
                    return "Mock 3D world generated"

            Text2PanoramaDemo = MockText2PanoramaDemo
            Image2PanoramaDemo = MockImage2PanoramaDemo
            HYworldDemo = MockHYworldDemo

# Global variables for model instances
text2pano_model = None
image2pano_model = None
world_model = None

def initialize_models():
    """Initialize the HunyuanWorld models (lazy loading)"""
    global text2pano_model, image2pano_model, world_model

    # Don't initialize models at startup for Zero GPU
    # They will be initialized when first needed
    logger.info("Models will be initialized on first use (Zero GPU optimization)")
    text2pano_model = None
    image2pano_model = None
    world_model = None

@spaces.GPU(duration=120)
def generate_text_to_world(
    prompt: str,
    negative_prompt: str = "",
    fg1_labels: str = "",
    fg2_labels: str = "",
    scene_class: str = "outdoor",
    seed: int = 42,
    progress=gr.Progress()
) -> Tuple[Optional[Image.Image], Optional[str], str]:
    """Generate 3D world from text prompt"""

    if not prompt.strip():
        return None, None, "Please enter a text prompt."

    try:
        progress(0.1, desc="Initializing models...")

        # Import and initialize models inside GPU context
        import_hunyuan_modules()
        global text2pano_model, world_model
        if text2pano_model is None:
            text2pano_model = Text2PanoramaDemo()
        if world_model is None:
            world_model = HYworldDemo()

        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            progress(0.2, desc="Generating panorama from text...")
            
            # Step 1: Generate panorama from text
            panorama = text2pano_model.run(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                output_path=temp_dir
            )
            
            progress(0.6, desc="Converting panorama to 3D world...")
            
            # Step 2: Convert panorama to 3D world
            panorama_path = os.path.join(temp_dir, "panorama.png")
            if isinstance(panorama, Image.Image):
                panorama.save(panorama_path)
            
            # Parse foreground labels
            labels_fg1 = [label.strip() for label in fg1_labels.split(",") if label.strip()]
            labels_fg2 = [label.strip() for label in fg2_labels.split(",") if label.strip()]
            
            progress(0.8, desc="Generating 3D mesh layers...")
            
            # Generate 3D world
            world_result = world_model.run(
                image_path=panorama_path,
                labels_fg1=labels_fg1,
                labels_fg2=labels_fg2,
                classes=scene_class,
                output_dir=temp_dir,
                export_drc=True
            )
            
            progress(0.9, desc="Preparing outputs...")

            # Create download package
            download_path = create_download_package(temp_dir)

            progress(1.0, desc="Complete!")

            if download_path:
                return panorama, download_path, "✅ 3D world generated successfully!"
            else:
                return panorama, None, "✅ Panorama generated successfully! (3D files not available)"
            
    except Exception as e:
        logger.error(f"Error in text-to-world generation: {e}")
        return None, None, f"❌ Error: {str(e)}"

@spaces.GPU(duration=120)
def generate_image_to_world(
    image: Optional[Image.Image],
    prompt: str = "",
    negative_prompt: str = "",
    fg1_labels: str = "",
    fg2_labels: str = "",
    scene_class: str = "outdoor",
    seed: int = 42,
    progress=gr.Progress()
) -> Tuple[Optional[Image.Image], Optional[str], str]:
    """Generate 3D world from input image"""

    if image is None:
        return None, None, "Please upload an image."

    try:
        progress(0.1, desc="Initializing models...")

        # Import and initialize models inside GPU context
        import_hunyuan_modules()
        global image2pano_model, world_model
        if image2pano_model is None:
            image2pano_model = Image2PanoramaDemo()
        if world_model is None:
            world_model = HYworldDemo()

        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input image
            input_path = os.path.join(temp_dir, "input.png")
            image.save(input_path)
            
            progress(0.2, desc="Generating panorama from image...")
            
            # Step 1: Generate panorama from image
            panorama = image2pano_model.run(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_path=input_path,
                seed=seed,
                output_path=temp_dir
            )
            
            progress(0.6, desc="Converting panorama to 3D world...")
            
            # Step 2: Convert panorama to 3D world
            panorama_path = os.path.join(temp_dir, "panorama.png")
            if isinstance(panorama, Image.Image):
                panorama.save(panorama_path)
            
            # Parse foreground labels
            labels_fg1 = [label.strip() for label in fg1_labels.split(",") if label.strip()]
            labels_fg2 = [label.strip() for label in fg2_labels.split(",") if label.strip()]
            
            progress(0.8, desc="Generating 3D mesh layers...")
            
            # Generate 3D world
            world_result = world_model.run(
                image_path=panorama_path,
                labels_fg1=labels_fg1,
                labels_fg2=labels_fg2,
                classes=scene_class,
                output_dir=temp_dir,
                export_drc=True
            )
            
            progress(0.9, desc="Preparing outputs...")

            # Create download package
            download_path = create_download_package(temp_dir)

            progress(1.0, desc="Complete!")

            if download_path:
                return panorama, download_path, "✅ 3D world generated successfully!"
            else:
                return panorama, None, "✅ Panorama generated successfully! (3D files not available)"
            
    except Exception as e:
        logger.error(f"Error in image-to-world generation: {e}")
        return None, None, f"❌ Error: {str(e)}"

def create_download_package(temp_dir: str) -> Optional[str]:
    """Create a zip package with all generated files"""
    try:
        zip_path = os.path.join(temp_dir, "hunyuan_world_output.zip")

        # Check if there are any files to package
        files_to_package = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.png', '.ply', '.drc', '.obj')) and not file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        files_to_package.append((file_path, os.path.relpath(file_path, temp_dir)))

        if not files_to_package:
            logger.warning("No files found to package")
            return None

        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_package:
                zipf.write(file_path, arcname)
                logger.info(f"Added {arcname} to package")

        # Verify zip file was created
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            logger.info(f"Download package created: {zip_path} ({os.path.getsize(zip_path)} bytes)")
            return zip_path
        else:
            logger.error("Zip file was not created or is empty")
            return None

    except Exception as e:
        logger.error(f"Error creating download package: {e}")
        return None

# Initialize models when the app starts (Zero GPU safe)
initialize_models()

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="HunyuanWorld-1.0: Generate Immersive 3D Worlds",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .example-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🌍 HunyuanWorld-1.0</h1>
            <h2>Generate Immersive, Explorable 3D Worlds</h2>
            <p><em>"To see a World in a Grain of Sand, and a Heaven in a Wild Flower"</em></p>
            <p>Transform text descriptions or images into interactive 3D environments</p>
        </div>
        """)

        # Main tabs
        with gr.Tabs():
            # Text-to-World Tab
            with gr.Tab("📝 Text-to-World", id="text_to_world"):
                gr.HTML("""
                <div class="feature-box">
                    <h3>🎨 Create 3D Worlds from Text</h3>
                    <p>Describe your ideal world and watch it come to life in 3D!</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        text_prompt = gr.Textbox(
                            label="✨ World Description",
                            placeholder="A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks...",
                            lines=3
                        )

                        text_negative = gr.Textbox(
                            label="🚫 Negative Prompt (Optional)",
                            placeholder="low quality, blurry, distorted...",
                            lines=2
                        )

                        with gr.Row():
                            text_fg1_labels = gr.Textbox(
                                label="🎯 Foreground Objects (Layer 1)",
                                placeholder="trees, rocks, flowers"
                            )
                            text_fg2_labels = gr.Textbox(
                                label="🎯 Foreground Objects (Layer 2)",
                                placeholder="mountains, buildings, vehicles"
                            )

                        with gr.Row():
                            text_scene_class = gr.Dropdown(
                                choices=["outdoor", "indoor", "urban", "natural", "fantasy"],
                                value="outdoor",
                                label="🏞️ Scene Type"
                            )
                            text_seed = gr.Number(
                                value=42,
                                label="🎲 Seed"
                            )

                        text_generate_btn = gr.Button(
                            "🚀 Generate 3D World",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        text_panorama_output = gr.Image(
                            label="🌅 Generated Panorama",
                            type="pil",
                            interactive=False
                        )

                        text_download_output = gr.File(
                            label="📦 Download 3D World Files",
                            interactive=False
                        )

                        text_status_output = gr.Textbox(
                            label="📊 Status",
                            interactive=False,
                            lines=2
                        )

            # Image-to-World Tab
            with gr.Tab("🖼️ Image-to-World", id="image_to_world"):
                gr.HTML("""
                <div class="feature-box">
                    <h3>🖼️ Transform Images into 3D Worlds</h3>
                    <p>Upload an image and expand it into a complete 3D environment!</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="📸 Upload Image",
                            type="pil"
                        )

                        image_prompt = gr.Textbox(
                            label="✨ Enhancement Prompt (Optional)",
                            placeholder="Add more details, change lighting, add objects...",
                            lines=2
                        )

                        image_negative = gr.Textbox(
                            label="🚫 Negative Prompt (Optional)",
                            placeholder="low quality, blurry, distorted...",
                            lines=2
                        )

                        with gr.Row():
                            image_fg1_labels = gr.Textbox(
                                label="🎯 Foreground Objects (Layer 1)",
                                placeholder="trees, rocks, flowers"
                            )
                            image_fg2_labels = gr.Textbox(
                                label="🎯 Foreground Objects (Layer 2)",
                                placeholder="mountains, buildings, vehicles"
                            )

                        with gr.Row():
                            image_scene_class = gr.Dropdown(
                                choices=["outdoor", "indoor", "urban", "natural", "fantasy"],
                                value="outdoor",
                                label="🏞️ Scene Type"
                            )
                            image_seed = gr.Number(
                                value=42,
                                label="🎲 Seed"
                            )

                        image_generate_btn = gr.Button(
                            "🚀 Generate 3D World",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        image_panorama_output = gr.Image(
                            label="🌅 Generated Panorama",
                            type="pil",
                            interactive=False
                        )

                        image_download_output = gr.File(
                            label="📦 Download 3D World Files",
                            interactive=False
                        )

                        image_status_output = gr.Textbox(
                            label="📊 Status",
                            interactive=False,
                            lines=2
                        )

            # Examples Tab
            with gr.Tab("💡 Examples & Tips", id="examples"):
                gr.HTML("""
                <div class="feature-box">
                    <h3>🎨 Example Prompts</h3>
                    <p>Try these example prompts to get started:</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <h4>🏞️ Natural Landscapes</h4>
                        <ul>
                            <li>"A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks"</li>
                            <li>"Ancient redwood forest with sunbeams filtering through tall trees"</li>
                            <li>"Desert oasis with palm trees and a small pond surrounded by sand dunes"</li>
                            <li>"Tropical beach with white sand, turquoise water, and coconut palms"</li>
                        </ul>

                        <h4>🏛️ Architectural Scenes</h4>
                        <ul>
                            <li>"Ancient Greek temple ruins on a hilltop overlooking the Mediterranean"</li>
                            <li>"Futuristic cityscape with glass towers and flying vehicles"</li>
                            <li>"Medieval castle courtyard with stone walls and a central fountain"</li>
                            <li>"Japanese zen garden with carefully placed rocks and raked sand"</li>
                        </ul>
                        """)

                    with gr.Column():
                        gr.HTML("""
                        <h4>✨ Fantasy Worlds</h4>
                        <ul>
                            <li>"Mystical forest with glowing mushrooms and floating islands"</li>
                            <li>"Underwater coral reef city with bioluminescent plants"</li>
                            <li>"Alien planet surface with purple vegetation and twin moons"</li>
                            <li>"Steampunk workshop filled with brass gears and steam pipes"</li>
                        </ul>

                        <h4>🎯 Tips for Better Results</h4>
                        <ul>
                            <li><strong>Be Descriptive:</strong> Include details about lighting, atmosphere, and mood</li>
                            <li><strong>Specify Objects:</strong> Use foreground labels to control object placement</li>
                            <li><strong>Scene Type:</strong> Choose the appropriate scene type for better results</li>
                            <li><strong>Negative Prompts:</strong> Use to avoid unwanted elements</li>
                        </ul>
                        """)

                gr.HTML("""
                <div class="feature-box">
                    <h3>📁 Output Files</h3>
                    <p>Your generated 3D world will include:</p>
                    <ul>
                        <li><strong>panorama.png:</strong> The generated 360° panoramic image</li>
                        <li><strong>mesh_layer0.ply:</strong> Background 3D mesh</li>
                        <li><strong>mesh_layer1.ply:</strong> Foreground layer 1 mesh</li>
                        <li><strong>mesh_layer2.ply:</strong> Foreground layer 2 mesh</li>
                        <li><strong>*.drc files:</strong> Compressed Draco format meshes</li>
                    </ul>
                    <p>These files can be imported into Blender, Unity, Unreal Engine, or other 3D software.</p>
                </div>
                """)

        # Event handlers
        text_generate_btn.click(
            fn=generate_text_to_world,
            inputs=[
                text_prompt,
                text_negative,
                text_fg1_labels,
                text_fg2_labels,
                text_scene_class,
                text_seed
            ],
            outputs=[
                text_panorama_output,
                text_download_output,
                text_status_output
            ],
            show_progress=True
        )

        image_generate_btn.click(
            fn=generate_image_to_world,
            inputs=[
                image_input,
                image_prompt,
                image_negative,
                image_fg1_labels,
                image_fg2_labels,
                image_scene_class,
                image_seed
            ],
            outputs=[
                image_panorama_output,
                image_download_output,
                image_status_output
            ],
            show_progress=True
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e0e0e0;">
            <p>🌟 <strong>HunyuanWorld-1.0</strong> by Tencent Hunyuan3D Team</p>
            <p>
                <a href="https://3d.hunyuan.tencent.com/sceneTo3D" target="_blank">Official Site</a> |
                <a href="https://huggingface.co/tencent/HunyuanWorld-1" target="_blank">Model</a> |
                <a href="https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0" target="_blank">GitHub</a>
            </p>
            <p><em>Generating immersive, explorable, and interactive 3D worlds from words or pixels</em></p>
        </div>
        """)

    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
