# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Wan transformer models.

Tests cover:
- Model structure and instantiation (T2V and I2V)
- Forward pass sanity checks (T2V and I2V)
- Numerical correctness vs HuggingFace (T2V and I2V)
"""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

WAN_1_3B_CONFIG = {
    "attention_head_dim": 128,
    "eps": 1e-06,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "in_channels": 16,
    "num_attention_heads": 12,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "text_dim": 4096,
    "torch_dtype": "bfloat16",
    "cross_attn_norm": True,
    "hidden_size": 1536,
}


class TestWanTransformer(unittest.TestCase):
    """Unit tests for Wan transformer models."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_model_config(
        self, config_dict: dict, backend: str = "VANILLA"
    ) -> DiffusionModelConfig:
        """Create DiffusionModelConfig from config dict."""
        pretrained_config = SimpleNamespace(**config_dict)
        return DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
            skip_create_weights_in_init=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_model_structure(self):
        """Test Wan T2V model can be instantiated with correct structure."""
        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        config = deepcopy(WAN_1_3B_CONFIG)
        config["num_layers"] = 1

        model_config = self._create_model_config(config)
        model = WanTransformer3DModel(model_config=model_config)

        self.assertTrue(hasattr(model, "blocks"))
        self.assertEqual(len(model.blocks), 1)
        self.assertTrue(hasattr(model, "rope"))
        self.assertTrue(hasattr(model, "patch_embedding"))
        self.assertTrue(hasattr(model, "condition_embedder"))
        self.assertTrue(hasattr(model, "proj_out"))
        self.assertTrue(hasattr(model, "norm_out"))

        param_names = [n for n, _ in model.named_parameters()]
        self.assertTrue(any("ffn.up_proj" in n for n in param_names))
        self.assertTrue(any("ffn.down_proj" in n for n in param_names))

        block = model.blocks[0]
        self.assertTrue(hasattr(block, "attn1"))
        self.assertTrue(hasattr(block, "attn2"))
        self.assertIsNone(block.add_k_proj)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_i2v_model_structure(self):
        """Test Wan I2V model has image K/V cross-attention structure."""
        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        config = deepcopy(WAN_1_3B_CONFIG)
        config["num_layers"] = 1
        config["image_dim"] = 1280
        config["added_kv_proj_dim"] = 1536  # must equal hidden_size
        model_config = self._create_model_config(config)
        model = WanTransformer3DModel(model_config=model_config)

        block = model.blocks[0]
        self.assertIsNotNone(block.add_k_proj)
        self.assertIsNotNone(block.add_v_proj)
        self.assertIsNotNone(block.norm_added_k)
        self.assertIsNotNone(model.condition_embedder.image_embedder)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_forward_sanity(self):
        """Test Wan T2V forward pass produces valid output."""
        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        config = deepcopy(WAN_1_3B_CONFIG)
        config["num_layers"] = 1

        model_config = self._create_model_config(config)
        model = (
            WanTransformer3DModel(model_config=model_config)
            .to(self.DEVICE, dtype=torch.bfloat16)
            .eval()
        )

        batch_size = 1
        hidden_states = torch.randn(
            batch_size,
            config["in_channels"],
            1,
            8,
            8,
            device=self.DEVICE,
            dtype=torch.bfloat16,
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            32,
            config["text_dim"],
            device=self.DEVICE,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        # Note: With random weights, NaN can occur. For unit tests, we primarily check shapes.
        # Full numerical correctness is tested in TestWanHuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_i2v_forward_sanity(self):
        """Test Wan I2V forward pass produces valid output."""
        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        config = {
            "attention_head_dim": 128,
            "eps": 1e-06,
            "ffn_dim": 512,
            "freq_dim": 256,
            "in_channels": 16,
            "num_attention_heads": 4,
            "num_layers": 1,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 1024,
            "text_dim": 4096,
            "torch_dtype": "bfloat16",
            "cross_attn_norm": True,
            "hidden_size": 512,
            "image_dim": 1280,
            "added_kv_proj_dim": 512,
        }
        model_config = self._create_model_config(config)
        model = (
            WanTransformer3DModel(model_config=model_config)
            .to(self.DEVICE, dtype=torch.bfloat16)
            .eval()
        )

        batch_size = 1
        hidden_states = torch.randn(
            batch_size, 16, 1, 8, 8, device=self.DEVICE, dtype=torch.bfloat16
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.long)

        encoder_hidden_states = torch.randn(
            batch_size, 512, 4096, device=self.DEVICE, dtype=torch.bfloat16
        )
        encoder_hidden_states_image = torch.randn(
            batch_size, 4, 1280, device=self.DEVICE, dtype=torch.bfloat16
        )

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
            )

        self.assertEqual(output.shape, hidden_states.shape)


class TestWanHuggingFaceComparison(unittest.TestCase):
    """Test Wan models match HuggingFace reference implementation."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_model_config(self, config_dict: dict) -> DiffusionModelConfig:
        """Create DiffusionModelConfig from config dict."""
        pretrained_config = SimpleNamespace(**config_dict)
        return DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend="VANILLA"),
            skip_create_weights_in_init=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_i2v_allclose_to_hf(self):
        """Test TRT-LLM Wan I2V transformer matches HuggingFace output with a nonzero image input.

        Uses a tiny 1-layer I2V config with shared random weights. Passes a concrete
        nonzero image embedding to both models and checks TRT-LLM matches HF (cos_sim > 0.99).
        This directly tests the I2V image cross-attention path (add_k_proj / add_v_proj /
        norm_added_k / to_out) is computed correctly.
        The key concern is the non-trivial weight remapping in load_weights:
          blocks.N.attn2.add_k_proj  ->  blocks.N.add_k_proj
          blocks.N.attn2.add_v_proj  ->  blocks.N.add_v_proj
          blocks.N.attn2.norm_added_k  ->  blocks.N.norm_added_k
        A mismatch here would produce low cosine similarity despite numerically
        identical attention kernels.
        """
        try:
            from diffusers import WanTransformer3DModel as HFWanTransformer3DModel
        except ImportError:
            self.skipTest("diffusers not installed")

        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        torch.manual_seed(42)

        # Tiny I2V config: 1 layer, small hidden size, explicit image cross-attention
        hidden_size = 512
        config = {
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
            "hidden_size": hidden_size,
            "qk_norm": "rms_norm_across_heads",
            "cross_attn_norm": True,
            "image_dim": 1280,
            "added_kv_proj_dim": hidden_size,
        }
        dtype = torch.bfloat16

        hf_model = (
            HFWanTransformer3DModel(
                patch_size=config["patch_size"],
                num_attention_heads=config["num_attention_heads"],
                attention_head_dim=config["attention_head_dim"],
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                text_dim=config["text_dim"],
                freq_dim=config["freq_dim"],
                ffn_dim=config["ffn_dim"],
                num_layers=config["num_layers"],
                cross_attn_norm=config["cross_attn_norm"],
                qk_norm=config["qk_norm"],
                eps=config["eps"],
                image_dim=config["image_dim"],
                added_kv_proj_dim=config["added_kv_proj_dim"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        model_config = self._create_model_config(config)
        trtllm_model = (
            WanTransformer3DModel(model_config=model_config).to(self.DEVICE, dtype=dtype).eval()
        )

        trtllm_model.load_weights(hf_model.state_dict())

        batch_size = 1
        generator = torch.Generator(device=self.DEVICE).manual_seed(42)
        hidden_states = torch.randn(
            batch_size,
            config["in_channels"],
            1,
            8,
            8,
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            512,
            config["text_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        # Nonzero image embedding (seeded randn) — exercises add_k_proj / add_v_proj /
        # norm_added_k / to_out, the I2V-specific cross-attention path.
        encoder_hidden_states_image = torch.randn(
            batch_size,
            4,
            config["image_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )

        with (
            torch.no_grad(),
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ),
        ):
            hf_output = hf_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
            )

        hf_output = hf_output.float()
        trtllm_output = trtllm_output.float()

        cos_sim = F.cosine_similarity(
            hf_output.flatten().unsqueeze(0),
            trtllm_output.flatten().unsqueeze(0),
        ).item()
        max_diff = (hf_output - trtllm_output).abs().max().item()

        print("\n[Wan I2V HF Comparison]")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max diff: {max_diff:.6f}")

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim:.6f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wan_allclose_to_hf(self):
        """Test TRT-LLM Wan transformer matches HuggingFace output."""
        try:
            from diffusers import WanTransformer3DModel as HFWanTransformer3DModel
        except ImportError:
            self.skipTest("diffusers not installed")

        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        torch.manual_seed(42)

        config = deepcopy(WAN_1_3B_CONFIG)
        config["num_layers"] = 1
        dtype = torch.bfloat16

        hf_model = (
            HFWanTransformer3DModel(
                patch_size=config["patch_size"],
                num_attention_heads=config["num_attention_heads"],
                attention_head_dim=config["attention_head_dim"],
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                text_dim=config["text_dim"],
                freq_dim=config["freq_dim"],
                ffn_dim=config["ffn_dim"],
                num_layers=config["num_layers"],
                cross_attn_norm=config["cross_attn_norm"],
                qk_norm=config["qk_norm"],
                eps=config["eps"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        model_config = self._create_model_config(config)
        trtllm_model = (
            WanTransformer3DModel(model_config=model_config).to(self.DEVICE, dtype=dtype).eval()
        )

        trtllm_model.load_weights(hf_model.state_dict())

        batch_size = 1
        generator = torch.Generator(device=self.DEVICE).manual_seed(42)
        hidden_states = torch.randn(
            batch_size,
            config["in_channels"],
            1,
            8,
            8,
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        timestep = torch.tensor([500], device=self.DEVICE, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            32,
            config["text_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )

        with (
            torch.no_grad(),
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ),
        ):
            hf_output = hf_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        hf_output = hf_output.float()
        trtllm_output = trtllm_output.float()

        cos_sim = F.cosine_similarity(
            hf_output.flatten().unsqueeze(0),
            trtllm_output.flatten().unsqueeze(0),
        ).item()
        max_diff = (hf_output - trtllm_output).abs().max().item()

        print("\n[Wan HF Comparison]")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max diff: {max_diff:.6f}")

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
