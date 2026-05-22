# Visual Generation Examples

Quick reference for running visual generation models.
Please refer to [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
about the details of the feature.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| [`models/`](models/) | Per-model example scripts — slim API examples (~40 lines) that focus on model-specific request construction and output processing |
| [`configs/`](configs/) | YAML configs shared by offline examples (`--extra_visual_gen_options`) and `trtllm-serve` |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and client examples |

## Quick Start

[`quickstart_example.py`](quickstart_example.py) — generate a video in ~30 lines (Wan T2V, 1 GPU).

## Per-Model Examples

Each script under `models/` demonstrates a single model with the VisualGen API.
Engine config (quantization, parallelism, TeaCache, etc.) is an optional YAML
file passed via `--extra_visual_gen_options` — the same flag that `trtllm-serve` uses.

```bash
# Default: 1 GPU, model defaults
python models/wan_t2v.py

# With a shared config for NVFP4 quantization
python models/wan_t2v.py --extra_visual_gen_options configs/wan2.2-t2v-fp4-1gpu.yaml
```

## Prerequisites

```bash
# Install dependencies (from repository root)
pip install -r requirements-dev.txt
```


## FLUX (Text-to-Image)

### Basic Usage

**FLUX.1:**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --prompt "A cat sitting on a windowsill" \
    --height 1024 --width 1024 \
    --guidance_scale 3.5 \
    --output_path output.png
```

**With FP8 quantization:**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.2-dev \
    --prompt "A cat sitting on a windowsill" \
    --linear_type trtllm-fp8-per-tensor \
    --output_path output_fp8.png
```

**Batch mode (multiple prompts from file):**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.txt \
    --output_dir results/ --seed 42
```


## WAN (Text-to-Video)

The `models/wan_t2v.py` script generates a video from a built-in cinematic prompt.
Engine config (quantization, parallelism, attention backend) is specified via a YAML file
passed to `--extra_visual_gen_options`. Available configs under `configs/`:

| Config file | Model | GPUs | Quantization |
|-------------|-------|------|-------------|
| `wan2.1-t2v-bf16-1gpu.yaml` | Wan2.1-T2V-1.3B | 1 | BF16 (baseline) |
| `wan2.2-t2v-fp8-1gpu.yaml` | Wan2.2-T2V-A14B | 1 | FP8 blockwise dynamic |
| `wan2.2-t2v-fp4-1gpu.yaml` | Wan2.2-T2V-A14B | 1 | NVFP4 dynamic |
| `wan2.2-t2v-fp4-4gpu.yaml` | Wan2.2-T2V-A14B | 4 | NVFP4, CFG + Ulysses (2×2) |
| `wan2.2-t2v-fp8-8gpu.yaml` | Wan2.2-T2V-A14B | 8 | FP8, CFG + Ulysses (2×4) |

### Single GPU

**Wan 2.1 BF16 (1.3B, 480p):**
```bash
cd examples/visual_gen
python models/wan_t2v.py \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --extra_visual_gen_options configs/wan2.1-t2v-bf16-1gpu.yaml \
    --output_path output.mp4
```

**Wan 2.2 FP8 blockwise dynamic (14B, 720p):**
```bash
cd examples/visual_gen
python models/wan_t2v.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --extra_visual_gen_options configs/wan2.2-t2v-fp8-1gpu.yaml \
    --output_path output.mp4
```

**Wan 2.2 NVFP4 dynamic (14B, 720p):**
```bash
cd examples/visual_gen
python models/wan_t2v.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --extra_visual_gen_options configs/wan2.2-t2v-fp4-1gpu.yaml \
    --output_path output.mp4
```

### Multi-GPU Parallelism

WAN supports two parallelism modes controlled via YAML config:
- **CFG Parallelism** (`parallel.dit_cfg_size`): Split positive/negative guidance across GPUs
- **Ulysses Sequence Parallelism** (`parallel.dit_ulysses_size`): Split sequence along head dimension; must divide the model's head count (12 for Wan2.2)

Total GPUs required = `dit_cfg_size × dit_ulysses_size`.

**4 GPUs — Wan 2.2 NVFP4, CFG + Ulysses (2×2):**
```bash
cd examples/visual_gen
torchrun --nproc_per_node=4 models/wan_t2v.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --extra_visual_gen_options configs/wan2.2-t2v-fp4-4gpu.yaml \
    --output_path output.mp4
```

**8 GPUs — Wan 2.2 FP8, CFG + Ulysses (2×4):**
```bash
cd examples/visual_gen
torchrun --nproc_per_node=8 models/wan_t2v.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --extra_visual_gen_options configs/wan2.2-t2v-fp8-8gpu.yaml \
    --output_path output.mp4
```

Custom parallelism: write a YAML with `parallel.dit_cfg_size` and `parallel.dit_ulysses_size`
and pass it via `--extra_visual_gen_options`.


## WAN (Image-to-Video)

The `models/wan_i2v.py` script generates a video from a built-in image (`cat_piano.png`)
and prompt. As with `wan_t2v.py`, engine config (quantization, parallelism, attention
backend) is specified via a YAML file passed to `--extra_visual_gen_options`.
Available configs under `configs/`:

| Config file | Model | GPUs | Quantization |
|-------------|-------|------|-------------|
| `wan2.1-i2v-bf16-1gpu.yaml` | Wan2.1-I2V-14B-480P | 1 | BF16 (baseline) |
| `wan2.2-i2v-bf16-1gpu.yaml` | Wan2.2-I2V-A14B    | 1 | BF16 (baseline) |

### Single GPU

**Wan 2.1 I2V BF16 (14B, 480p):**
```bash
cd examples/visual_gen
python models/wan_i2v.py \
    --model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --extra_visual_gen_options configs/wan2.1-i2v-bf16-1gpu.yaml \
    --output_path output.mp4
```

**Wan 2.2 I2V BF16 (A14B):**
```bash
cd examples/visual_gen
python models/wan_i2v.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --extra_visual_gen_options configs/wan2.2-i2v-bf16-1gpu.yaml \
    --output_path output.mp4
```

To use a different input image or prompt, edit the inline values in
[`models/wan_i2v.py`](models/wan_i2v.py).


## LTX2 (Text/Image-to-Video with Audio)

LTX2 generates video **with audio** from text prompts or input images.
It uses a Gemma3 text encoder (provided separately via `--text_encoder_path`)
and supports BF16, FP8, and FP4 precision checkpoints.

Please refer to tensorrt_llm/_torch/visual_gen/models/ltx2/LTX_2_CHECKPOINT_FORMAT.md for model checkpoint info.

### Basic Usage

**Text-to-Video (single GPU):**
```bash
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 720 --width 1280 --num_frames 121 \
    --steps 40 --guidance_scale 4.0 --seed 42 \
    --output_path output_t2v.mp4
```

**Image-to-Video:**
```bash
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --image ${PROJECT_ROOT}/examples/visual_gen/cat_piano.png \
    --image_cond_strength 1.0 \
    --height 720 --width 1280 --num_frames 121 \
    --steps 40 --seed 42 \
    --output_path output_i2v.mp4
```

### Precision Variants

LTX2 ships checkpoints at three precision levels. Simply point `--model_path` at the
appropriate directory:

```bash
# FP8
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/fp8/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 720 --width 1280 --num_frames 121 \
    --output_path output_fp8.mp4

# FP4
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/fp4/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 512 --width 768 --num_frames 121 \
    --output_path output_fp4.mp4
```

---

## Common Arguments

> **Note:** `models/wan_t2v.py` and `models/wan_i2v.py` use only `--model`,
> `--extra_visual_gen_options`, and `--output_path`. All other settings
> (quantization, parallelism, attention backend, TeaCache, etc.) are specified
> in the YAML config. The table below applies to `visual_gen_flux.py` and
> `visual_gen_ltx2.py`.

| Argument | FLUX | LTX2 | Default | Description |
|----------|------|------|---------|-------------|
| `--model_path` | ✓ | — | Path to model checkpoint directory |
| `--text_encoder_path` | — | ✓ | Path to Gemma3 text encoder |
| `--prompt` | ✓ | ✓ | Text prompt for generation |
| `--negative_prompt` | — | *(built-in)* | Negative prompt |
| `--height` | ✓ | ✓ | 1024 / 720 | Output height |
| `--width` | ✓ | ✓ | 1024 / 1280 | Output width |
| `--num_frames` | — | ✓ | 121 | Number of frames |
| `--frame_rate` | — | 24.0 | Output frame rate (fps) |
| `--steps` | ✓ | ✓ | 50 / 40 | Denoising steps |
| `--guidance_scale` | ✓ | ✓ | 3.5 / 4.0 | Guidance strength |
| `--seed` | ✓ | ✓ | 42 | Random seed |
| `--image` | — | ✓ | None | Input image for image-to-video |
| `--image_cond_strength` | — | ✓ | 1.0 | Image conditioning strength |
| `--enable_teacache` | ✓ | — | False | Cache optimization |
| `--teacache_thresh` | ✓ | — | 0.2 | TeaCache similarity threshold |
| `--attention_backend` | ✓ | — | VANILLA | `VANILLA`, `TRTLLM`, or `FA4` |
| `--enable_sage_attention` | ✓ | — | False | SageAttention (requires `TRTLLM` attention backend) |
| `--ulysses_size` | ✓ | — | 1 | Ulysses parallelism |
| `--attn2d_row_size` | ✓ | ✓ | 1 | Attention2D mesh row size |
| `--attn2d_col_size` | ✓ | ✓ | 1 | Attention2D mesh column size |
| `--linear_type` | ✓ | — | default | Quantization type |
| `--enhance_prompt` | — | ✓ | False | Gemma3 prompt enhancement |
| `--stg_scale` | — | ✓ | 0.0 | Spatiotemporal guidance scale |
| `--modality_scale` | — | ✓ | 1.0 | Cross-modal guidance scale |
| `--rescale_scale` | — | ✓ | 0.0 | Variance-preserving rescale factor |

For WAN T2V/I2V multi-GPU and quantization, see the WAN sections above — those
are controlled via YAML configs under `configs/`, not CLI flags.

## Troubleshooting

**Out of Memory:**
- WAN T2V / I2V: use a quantized or multi-GPU config (e.g., `configs/wan2.2-t2v-fp8-1gpu.yaml`,
  `configs/wan2.2-t2v-fp4-4gpu.yaml`) via `--extra_visual_gen_options`
- FLUX: use `--linear_type trtllm-fp8-per-tensor` or `--linear_type trtllm-fp8-blockwise`
- Reduce resolution or frames (edit `default_params` in
  `tensorrt_llm/_torch/visual_gen/models/wan/defaults.py` for WAN)
- Enable TeaCache via YAML config (`teacache.enable_teacache: true`) or `--enable_teacache` flag (FLUX)

**Slow Inference:**
- WAN T2V / I2V: use a multi-GPU config via `--extra_visual_gen_options` with `torchrun --nproc_per_node=N`
- FLUX: `--attention_backend TRTLLM`, `--cfg_size 2`, or `--ulysses_size 2`
- Enable TeaCache

**Import Errors:**
- Run from repository root
- Install necessary dependencies, e.g., `pip install -r requirements-dev.txt`

**Ulysses Errors:**
- `dit_ulysses_size` (WAN T2V/I2V YAML) or `--ulysses_size` (FLUX) must divide the model's head count (12 for Wan2.2)
- If your GPU count does not divide the head count, adjust `dit_cfg_size × dit_ulysses_size` accordingly
  (`--attention_backend FA4 --attn2d_row_size <row> --attn2d_col_size <col>`)
- Total GPUs = `cfg_size × ulysses_size`
- Sequence length must be divisible by `ulysses_size`

**Attention2D Errors:**
- Requires `--attention_backend FA4`
- Combining with `--ulysses_size` is not yet supported
- Total GPUs = `cfg_size × attn2d_row_size × attn2d_col_size`
- Sequence length must be divisible by `attn2d_row_size × attn2d_col_size`

## Output Formats

- **FLUX**: `.png` (image)
- **WAN**: `.mp4` if FFmpeg is installed, otherwise `.avi` (video)
- **LTX2**: `.mp4` (video with audio) if FFmpeg is installed, otherwise `.avi` (video)

## Serving

See [`serve/README.md`](serve/README.md) for `trtllm-serve` examples including image generation (FLUX), video generation (WAN T2V/I2V), and API endpoint reference.
