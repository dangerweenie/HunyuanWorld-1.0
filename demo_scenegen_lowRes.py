# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].

# ==========================================================================================
# This script provides a low-resolution and faster alternative for 3D world generation.
# It takes a panoramic image as input and converts it into a 3D mesh, separating
# foreground and background elements into different layers.
#
# Key Modifications for Performance:
# - Default Resolution: The internal processing resolution is reduced to 1920px wide
#   to decrease VRAM usage and speed up depth estimation and mesh generation.
# - Super-Resolution Disabled by Default: The script no longer uses super-resolution
#   on the input images by default. It can be re-enabled with the --use_sr flag.
#
# Usage:
# python demo_scenegen_lowRes.py --image_path /path/to/your/panorama.png --labels_fg1 "person"
# ==========================================================================================
import os
import torch
import open3d as o3d

import argparse

# hunyuan3d scene generation
from hy3dworld import LayerDecomposition
from hy3dworld import WorldComposer, process_file


class HYworldDemo:
    def __init__(self, seed=42):

        # MODIFICATION: Reduced target_size for lower resource consumption
        target_size = 1920
        kernel_scale = max(1, int(target_size / 1920))

        self.LayerDecomposer = LayerDecomposition()

        self.hy3d_world = WorldComposer(
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            resolution=(target_size, target_size // 2),
            seed=seed,
            filter_mask=True,
            kernel_scale=kernel_scale,
        )

    def run(self, image_path, labels_fg1, labels_fg2, classes="outdoor", output_dir='output_hyworld', export_drc=False, use_sr=False):
        # foreground layer information
        fg1_infos = [
            {
                "image_path": image_path,
                "output_path": output_dir,
                "labels": labels_fg1,
                "class": classes,
            }
        ]
        fg2_infos = [
            {
                "image_path": os.path.join(output_dir, 'remove_fg1_image.png'),
                "output_path": output_dir,
                "labels": labels_fg2,
                "class": classes,
            }
        ]

        # layer decompose
        self.LayerDecomposer(fg1_infos, layer=0)
        self.LayerDecomposer(fg2_infos, layer=1)
        self.LayerDecomposer(fg2_infos, layer=2)
        # MODIFICATION: Changed sr=True to sr=use_sr (defaults to False) to disable super-resolution
        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(
            output_dir, sr=use_sr
        )

        # layer-wise reconstruction
        layered_world_mesh = self.hy3d_world.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type='mesh'
        )

        # save results
        for layer_idx, layer_info in enumerate(layered_world_mesh):
            # export ply
            output_path = os.path.join(
                output_dir, f"mesh_layer{layer_idx}.ply"
            )
            o3d.io.write_triangle_mesh(output_path, layer_info['mesh'])

            # export drc
            if export_drc:
                output_path_drc = os.path.join(
                    output_dir, f"mesh_layer{layer_idx}.drc"
                )
                process_file(output_path, output_path_drc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D World Gen Demo (Low Resolution)")
    parser.add_argument("--image_path", type=str,
                        default=None, help="Path to the Panorama image")
    parser.add_argument("--labels_fg1", nargs='+', default=[],
                        help="Labels for foreground objects in layer 1")
    parser.add_argument("--labels_fg2", nargs='+', default=[],
                        help="Labels for foreground objects in layer 2")
    parser.add_argument("--classes", type=str, default="outdoor",
                        help="Classes for scene generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_path", type=str, default="results",
                        help="Path to save the output results")
    parser.add_argument("--export_drc", type=bool, default=False,
                        help="Whether to export Draco format")
    # MODIFICATION: Added command-line flag to optionally enable Super Resolution
    parser.add_argument("--use_sr", action='store_true',
                        help="Enable super-resolution for input images (increases resource usage)")
    
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}")

    demo_HYworld = HYworldDemo(seed=args.seed)
    demo_HYworld.run(
        image_path=args.image_path,
        labels_fg1=args.labels_fg1,
        labels_fg2=args.labels_fg2,
        classes=args.classes,
        output_dir=args.output_path,
        export_drc=args.export_drc,
        use_sr=args.use_sr
    )
