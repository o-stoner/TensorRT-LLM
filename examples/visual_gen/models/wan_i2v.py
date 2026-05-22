#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wan Image-to-Video generation.

Usage:
    python wan_i2v.py
    python wan_i2v.py --extra_visual_gen_options ../configs/wan2.2-i2v-bf16-1gpu.yaml
"""

import argparse
import os

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

_DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cat_piano.png")


def main():
    parser = argparse.ArgumentParser(description="Wan Image-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--extra_visual_gen_options",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --extra_visual_gen_options)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="wan_i2v_output.mp4",
        help="Path to save the output video",
    )
    args = parser.parse_args()

    extra_args = (
        VisualGenArgs.from_yaml(args.extra_visual_gen_options)
        if args.extra_visual_gen_options
        else None
    )
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: I2V request construction ---
    params = visual_gen.default_params

    output = visual_gen.generate(
        inputs=(
            "A cat playfully presses the piano keys with its paws, its head gently "
            "bobbing as soft melodic notes fill the warmly lit room, the camera slowly "
            "pulling back to reveal the cozy interior."
        ),
        params=VisualGenParams(
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            seed=params.seed,
            num_frames=params.num_frames,
            frame_rate=params.frame_rate,
            image=_DEFAULT_IMAGE,
        ),
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
