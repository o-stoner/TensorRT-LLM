# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Wan I2V pipeline.

Tests cover:
- Pipeline loading (Wan 2.1/2.2 instantiation, attention backend config)
- Quantization: FP8 weight verification
- FP8 vs BF16 numerical correctness (single layer + full transformer E2E)
- FP8 memory reduction (~2x vs BF16)
- End-to-end HuggingFace comparison (5 denoising steps, seed determinism)

"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionArgs,
    DiffusionModelConfig,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.models.modeling_utils import QuantConfig

_DEFAULT_I2V_CHECKPOINT = "/home/scratch.trt_llm_data_ci/llm-models/Wan2.2-I2V-A14B-Diffusers"


def _resolve_checkpoint_i2v() -> str:
    """
    Uses DIFFUSION_MODEL_PATH env var if set, otherwise falls back to the
    default CI checkpoint path.  Raises ValueError if the checkpoint exists
    but its model_index.json does not identify it as an I2V pipeline.
    """
    path = Path(os.environ.get("DIFFUSION_MODEL_PATH", _DEFAULT_I2V_CHECKPOINT))

    if not path.exists():
        return str(path)  # checkpoint absent; fixtures will skip

    model_index = path / "model_index.json"
    if model_index.exists():
        with open(model_index) as f:
            class_name = json.load(f).get("_class_name", "")
        if "ImageToVideo" not in class_name and "i2v" not in class_name.lower():
            raise ValueError(
                f"Checkpoint at {path} does not appear to be a Wan I2V model "
                f"(model_index.json _class_name={class_name!r}). "
                "Set DIFFUSION_MODEL_PATH to a Wan I2V checkpoint."
            )

    return str(path)


CHECKPOINT_PATH = _resolve_checkpoint_i2v()

SKIP_COMPONENTS = []


def _get_wan_i2v_transformer_inputs(transformer, device="cuda", dtype=torch.bfloat16):
    """Create test inputs for the Wan I2V transformer (H=W=64, T=1, seq_len=1024)."""
    torch.manual_seed(42)
    config = transformer.config
    in_channels = getattr(config, "in_channels", 16)
    image_dim = getattr(config, "image_dim", None)
    hidden_states = torch.randn(1, in_channels, 1, 64, 64, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 32, 4096, device=device, dtype=dtype)
    # image_dim is None for Wan 2.2 I2V (no CLIP embeddings); pass None in that case.
    encoder_hidden_states_image = (
        torch.randn(1, 4, image_dim, device=device, dtype=dtype) if image_dim is not None else None
    )
    return dict(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )


def _find_first_quantizable_linear(transformer):
    """Find the first quantizable Linear in transformer blocks."""
    if hasattr(transformer, "blocks") and len(transformer.blocks) > 0:
        block = transformer.blocks[0]
        if hasattr(block, "attn1") and hasattr(block.attn1, "qkv_proj"):
            return block.attn1.qkv_proj, "blocks.0.attn1.qkv_proj"
    for name, module in transformer.named_modules():
        if isinstance(module, Linear) and "blocks" in name:
            return module, name
    return None, None


@pytest.fixture
def checkpoint_exists():
    """Skip test if I2V checkpoint is unavailable."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(
            f"Wan I2V checkpoint not found at {CHECKPOINT_PATH}. "
            "Set DIFFUSION_MODEL_PATH or stage checkpoint under LLM_MODELS_ROOT."
        )
    return True


def _load_i2v_pipeline(quant_config=None):
    """Load Wan I2V pipeline inline. Caller must delete and empty_cache."""
    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_COMPONENTS,
        **({"quant_config": quant_config} if quant_config else {}),
    )
    return PipelineLoader(args).load()


class TestWanI2VPipelineLoading:
    """Integration tests for Wan I2V pipeline loading."""

    _TINY_I2V_CONFIG = {
        "attention_head_dim": 128,
        "eps": 1e-06,
        "ffn_dim": 512,
        "freq_dim": 256,
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 4,
        "num_layers": 1,
        "patch_size": [1, 2, 2],
        "text_dim": 4096,
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "qk_norm": "rms_norm_across_heads",
        "cross_attn_norm": True,
        "image_dim": 1280,
        "added_kv_proj_dim": 512,
    }

    @classmethod
    def _make_tiny_model_config(cls, boundary_ratio=None) -> DiffusionModelConfig:
        config_dict = dict(cls._TINY_I2V_CONFIG)
        if boundary_ratio is not None:
            config_dict["boundary_ratio"] = boundary_ratio
        return DiffusionModelConfig(
            pretrained_config=SimpleNamespace(**config_dict),
            quant_config=QuantConfig(),
            skip_create_weights_in_init=True,
        )

    def test_wan21_i2v_instantiation(self):
        """Test Wan 2.1 I2V pipeline (single-stage) instantiates correctly."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        pipeline = WanImageToVideoPipeline(self._make_tiny_model_config(boundary_ratio=None))
        assert pipeline.transformer is not None
        assert pipeline.transformer_2 is None
        assert pipeline.boundary_ratio is None

    def test_wan22_i2v_instantiation(self):
        """Test Wan 2.2 I2V pipeline (two-stage) instantiates correctly."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        pipeline = WanImageToVideoPipeline(self._make_tiny_model_config(boundary_ratio=0.4))
        assert pipeline.transformer is not None
        assert pipeline.transformer_2 is not None
        assert pipeline.boundary_ratio == 0.4

    @pytest.mark.parametrize(
        "backend",
        [
            "VANILLA",
            "TRTLLM",
        ],
    )
    def test_attention_backend_config(self, backend: str):
        """Verify AttentionConfig.backend threads through DiffusionModelConfig (no weights loaded)."""
        model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(**self._TINY_I2V_CONFIG),
            quant_config=QuantConfig(),
            attention=AttentionConfig(backend=backend),
            skip_create_weights_in_init=True,
        )
        assert model_config.attention.backend == backend


class TestWanI2VvsHF:
    """Compare TRT-LLM Wan I2V output against HuggingFace reference."""

    _PROMPT = "A bird flies over a tranquil lake"
    _NEGATIVE_PROMPT = ""
    _HEIGHT = 480
    _WIDTH = 832
    _NUM_FRAMES = 5
    _NUM_STEPS = 5
    _GUIDANCE = 5.0
    _SEED = 42

    @staticmethod
    def _make_ref_image(height: int, width: int):
        """Create a deterministic synthetic RGB image for I2V conditioning."""
        import numpy as np
        import PIL.Image

        rng = np.random.default_rng(0)
        pixels = rng.integers(64, 192, (height, width, 3), dtype=np.uint8)
        return PIL.Image.fromarray(pixels)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_e2e_vs_hf(self, checkpoint_exists):
        """TRT-LLM I2V pipeline output must closely match HuggingFace reference.

        Runs both pipelines sequentially (HF then TRT-LLM) at 480x832 / 5 frames /
        5 steps with the same seed and synthetic conditioning image, then asserts
        cosine similarity > 0.99.
        """
        pytest.importorskip(
            "ftfy", reason="ftfy required by diffusers WanImageToVideoPipeline prompt cleaning"
        )
        try:
            from diffusers import WanImageToVideoPipeline as HFWanI2VPipeline
        except ImportError:
            pytest.skip("diffusers WanImageToVideoPipeline not available")

        ref_image = self._make_ref_image(self._HEIGHT, self._WIDTH)

        # --- 1. HF pipeline ---
        gc.collect()
        torch.cuda.empty_cache()
        hf_pipe = HFWanI2VPipeline.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
        hf_pipe.to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(self._SEED)
        hf_out = hf_pipe(
            image=ref_image,
            prompt=self._PROMPT,
            negative_prompt=self._NEGATIVE_PROMPT,
            height=self._HEIGHT,
            width=self._WIDTH,
            num_frames=self._NUM_FRAMES,
            num_inference_steps=self._NUM_STEPS,
            guidance_scale=self._GUIDANCE,
            generator=generator,
            output_type="pt",
            return_dict=False,
        )
        # HF returns (frames,) where frames is (B, T, C, H, W) float [0, 1]
        hf_frames = hf_out[0][0].permute(0, 2, 3, 1).clamp(0, 1).float().cpu()  # (T, H, W, C)
        print(
            f"\n[HF]     shape={hf_frames.shape}  range=[{hf_frames.min():.3f}, {hf_frames.max():.3f}]"
        )

        del hf_pipe, hf_out
        gc.collect()
        torch.cuda.empty_cache()

        # --- 2. TRT-LLM pipeline ---
        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        trtllm_pipe = PipelineLoader(args).load()
        trtllm_result = trtllm_pipe.forward(
            image=ref_image,
            prompt=self._PROMPT,
            negative_prompt=self._NEGATIVE_PROMPT,
            height=self._HEIGHT,
            width=self._WIDTH,
            num_frames=self._NUM_FRAMES,
            num_inference_steps=self._NUM_STEPS,
            guidance_scale=self._GUIDANCE,
            seed=self._SEED,
        )
        # TRT-LLM returns MediaOutput(video=(T, H, W, C) uint8)
        trtllm_frames = trtllm_result.video.float().cpu() / 255.0  # (T, H, W, C) float [0, 1]
        print(
            f"[TRTLLM] shape={trtllm_frames.shape}  range=[{trtllm_frames.min():.3f}, {trtllm_frames.max():.3f}]"
        )

        del trtllm_pipe
        gc.collect()
        torch.cuda.empty_cache()

        # --- 3. Compare ---
        assert hf_frames.shape == trtllm_frames.shape, (
            f"Shape mismatch: HF={hf_frames.shape} vs TRTLLM={trtllm_frames.shape}"
        )
        cos_sim = F.cosine_similarity(
            hf_frames.flatten().unsqueeze(0), trtllm_frames.flatten().unsqueeze(0)
        ).item()
        print(f"[Compare] cosine similarity = {cos_sim:.6f}  (threshold 0.99)")
        assert cos_sim > 0.99, (
            f"TRT-LLM I2V output deviates too much from HF reference: cos_sim={cos_sim:.6f} < 0.99"
        )


@pytest.fixture(scope="class")
def fp8_refs():
    """Class-scoped fixture: load BF16 once to collect all reference data, unload,
    then load FP8 once and keep it alive for the duration of the test class.
    """
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(
            f"Wan I2V checkpoint not found at {CHECKPOINT_PATH}. "
            "Set DIFFUSION_MODEL_PATH or stage checkpoint under LLM_MODELS_ROOT."
        )

    refs = {}

    # --- Phase 1: BF16 (load once, collect all reference data, then unload) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pipeline_bf16 = _load_i2v_pipeline()
    try:
        transformer_bf16 = pipeline_bf16.transformer

        # Single-layer reference
        linear_bf16, layer_name = _find_first_quantizable_linear(transformer_bf16)
        assert linear_bf16 is not None, "No quantizable Linear found in BF16 transformer"
        weight_bf16 = linear_bf16.weight.data.clone()
        bias_bf16 = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None
        in_features = linear_bf16.in_features
        torch.manual_seed(42)
        input_tensor = torch.randn(1024, in_features, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            expected = F.linear(input_tensor, weight_bf16, bias_bf16)
            result_bf16_layer = linear_bf16(input_tensor)
        assert torch.allclose(result_bf16_layer, expected, rtol=1e-5, atol=1e-6), (
            "BF16 layer should match F.linear reference exactly"
        )
        refs["expected_cpu"] = expected.cpu()
        refs["in_features"] = in_features
        refs["layer_name"] = layer_name

        # Full E2E reference
        e2e_inputs = _get_wan_i2v_transformer_inputs(transformer_bf16)
        with torch.no_grad():
            output_bf16 = transformer_bf16(**e2e_inputs)
        bf16_f_cpu = output_bf16.float().cpu()
        assert not torch.isnan(bf16_f_cpu).any(), "BF16 output contains NaN"
        refs["bf16_f_cpu"] = bf16_f_cpu
        refs["output_shape"] = output_bf16.shape

        # Memory footprint
        refs["bf16_mem_gb"] = (
            sum(p.numel() * p.element_size() for p in transformer_bf16.parameters()) / 1024**3
        )
        refs["bf16_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        del pipeline_bf16
        gc.collect()
        torch.cuda.empty_cache()

    # --- Phase 2: FP8 (load once, stay alive for all tests in this class) ---
    torch.cuda.reset_peak_memory_stats()
    pipeline_fp8 = _load_i2v_pipeline(quant_config={"quant_algo": "FP8", "dynamic": True})
    refs["fp8_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    refs["pipeline_fp8"] = pipeline_fp8

    yield SimpleNamespace(**refs)

    del pipeline_fp8
    gc.collect()
    torch.cuda.empty_cache()


class TestWanI2VFP8:
    """FP8 quantization, numerical correctness, and memory tests for Wan I2V."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_weight_verification(self, fp8_refs):
        """Verify FP8 quantized weights are stored with correct dtype."""
        pipeline = fp8_refs.pipeline_fp8
        assert pipeline.model_config.quant_config.quant_algo is not None
        quant_count = 0
        found_quant = False
        for name, module in pipeline.transformer.named_modules():
            if (
                isinstance(module, Linear)
                and module.quant_config
                and module.quant_config.quant_algo
            ):
                quant_count += 1
                if "blocks" in name and not found_quant:
                    if hasattr(module, "weight") and module.weight is not None:
                        assert module.weight.dtype == torch.float8_e4m3fn, (
                            f"{name}: expected FP8 weight, got {module.weight.dtype}"
                        )
                        assert hasattr(module, "weight_scale"), f"{name} missing weight_scale"
                    found_quant = True
                    print(f"\n[FP8] First quant layer: {name}")
        assert quant_count > 0, "No layers were quantized for FP8"
        assert found_quant, "No quantized Linear modules found in blocks for FP8"
        print(f"[FP8] Quantized {quant_count} Linear layers")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_single_layer(self, fp8_refs):
        """Test FP8 vs BF16 accuracy on the first quantizable Linear layer."""
        layer_name = fp8_refs.layer_name
        in_features = fp8_refs.in_features

        linear_fp8, _ = _find_first_quantizable_linear(fp8_refs.pipeline_fp8.transformer)
        assert linear_fp8 is not None, "No quantizable Linear found in FP8 transformer"

        torch.manual_seed(42)
        input_tensor_fp8 = torch.randn(1024, in_features, dtype=torch.bfloat16, device="cuda")
        print(f"\n[FP8 Single Layer] Layer: {layer_name}, input: {input_tensor_fp8.shape}")

        with torch.no_grad():
            result_fp8 = linear_fp8(input_tensor_fp8)

        expected_gpu = fp8_refs.expected_cpu.to("cuda")
        cos_sim = F.cosine_similarity(
            result_fp8.flatten().float(), expected_gpu.flatten().float(), dim=0
        )
        mse = F.mse_loss(result_fp8.flatten().float(), expected_gpu.flatten().float())
        max_diff = (result_fp8 - expected_gpu).abs().max().item()

        print(f"  max_diff={max_diff:.6f}, cos_sim={cos_sim.item():.6f}, mse={mse.item():.6f}")
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"
        assert mse < 1.0, f"MSE too high: {mse.item()}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_full_transformer_e2e(self, fp8_refs):
        """End-to-end test: Compare full Wan I2V transformer FP8 vs BF16 output."""
        inputs_fp8 = _get_wan_i2v_transformer_inputs(fp8_refs.pipeline_fp8.transformer)
        with torch.no_grad():
            output_fp8 = fp8_refs.pipeline_fp8.transformer(**inputs_fp8)

        assert output_fp8.shape == fp8_refs.output_shape, (
            f"Shape mismatch: BF16={fp8_refs.output_shape}, FP8={output_fp8.shape}"
        )
        fp8_f = output_fp8.float()
        assert not torch.isnan(fp8_f).any(), "FP8 output contains NaN"
        assert not torch.isinf(fp8_f).any(), "FP8 output contains Inf"

        bf16_f = fp8_refs.bf16_f_cpu.to("cuda")
        cos_sim = F.cosine_similarity(fp8_f.flatten(), bf16_f.flatten(), dim=0).item()
        mean_diff = (fp8_f - bf16_f).abs().mean().item()
        rel_error = mean_diff / (bf16_f.abs().mean().item() + 1e-8)

        print(
            f"\n[FP8 E2E] BF16 range: [{fp8_refs.bf16_f_cpu.min():.4f}, {fp8_refs.bf16_f_cpu.max():.4f}]"
        )
        print(f"[FP8 E2E] cos_sim={cos_sim:.6f}, rel_error={rel_error:.6f}")
        print(f"  FP8  range: [{fp8_f.min():.4f}, {fp8_f.max():.4f}]")
        assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim:.6f} (expected >0.95)"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_vs_bf16_memory_comparison(self, fp8_refs):
        """Test FP8 transformer uses ~2x less parameter memory than BF16."""
        bf16_mem = fp8_refs.bf16_mem_gb
        fp8_mem = (
            sum(
                p.numel() * p.element_size() for p in fp8_refs.pipeline_fp8.transformer.parameters()
            )
            / 1024**3
        )
        ratio = bf16_mem / fp8_mem
        peak_ratio = fp8_refs.bf16_peak_mem_gb / fp8_refs.fp8_peak_mem_gb
        print(f"\n[Memory] BF16: {bf16_mem:.2f} GB, FP8: {fp8_mem:.2f} GB, ratio: {ratio:.2f}x")
        print(
            f"[Peak]   BF16: {fp8_refs.bf16_peak_mem_gb:.2f} GB, "
            f"FP8: {fp8_refs.fp8_peak_mem_gb:.2f} GB, ratio: {peak_ratio:.2f}x"
        )
        assert ratio > 1.8, f"FP8 should use ~2x less memory, got {ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
