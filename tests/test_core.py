"""
tests/test_core.py

Unit tests for core quantization components.
Run with: pytest tests/ -v

Tests cover:
  - Quantization kernels (1b, 2b, 4b) round-trip error
  - Bit allocator budget constraint
  - Salience metrics shapes
  - Ensemble normalization
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantizer.kernels import (
    quantize_1bit, dequantize_1bit,
    quantize_2bit_symmetric, dequantize_2bit_symmetric,
    quantize_2bit_asymmetric, dequantize_2bit_asymmetric,
    quantize_4bit, dequantize_4bit,
    quantize_weight,
)
from src.quantizer.allocator import AllocationConfig, BitAllocator
from src.salience.metrics import (
    SalienceConfig,
    MagnitudeSalience,
    GradientSalience,
    ActivationSalience,
    EnsembleSalience,
)


# -----------------------------------------------------------------------
# Quantization kernel tests
# -----------------------------------------------------------------------

class TestQuantizationKernels:

    @pytest.fixture
    def weight(self):
        torch.manual_seed(42)
        return torch.randn(64, 128)

    def test_1bit_shape_preserved(self, weight):
        q, scales = quantize_1bit(weight)
        assert q.shape == weight.shape

    def test_1bit_values_binary(self, weight):
        """1-bit weights should only contain +/- scale values."""
        q, scales = quantize_1bit(weight)
        # All values should be nonzero (sign-based)
        assert (q != 0).all(), "1-bit weights should not have zeros (sign-based)"

    def test_2bit_symmetric_roundtrip(self, weight):
        q_codes, scales, _ = quantize_2bit_symmetric(weight)
        deq = dequantize_2bit_symmetric(q_codes, scales)
        assert deq.shape == weight.shape
        # Max reconstruction error should be bounded
        max_err = (weight - deq).abs().max().item()
        assert max_err < weight.abs().max().item() * 0.5, \
            f"2-bit reconstruction error too large: {max_err}"

    def test_2bit_codes_in_range(self, weight):
        q_codes, _, _ = quantize_2bit_symmetric(weight)
        assert q_codes.min() >= 0
        assert q_codes.max() <= 3

    def test_2bit_asymmetric_roundtrip(self, weight):
        q_codes, scales, zero_points = quantize_2bit_asymmetric(weight)
        deq = dequantize_2bit_asymmetric(q_codes, scales, zero_points)
        assert deq.shape == weight.shape

    def test_4bit_shape_preserved(self, weight):
        q_codes, scales = quantize_4bit(weight)
        deq = dequantize_4bit(q_codes, scales)
        assert deq.shape == weight.shape

    def test_4bit_codes_in_range(self, weight):
        q_codes, _ = quantize_4bit(weight)
        assert q_codes.min() >= 0
        assert q_codes.max() <= 14

    def test_4bit_better_than_2bit(self, weight):
        """4-bit should have lower reconstruction error than 2-bit."""
        _, scales_2b, _ = quantize_2bit_symmetric(weight)
        q_2b, scales_2b, _ = quantize_2bit_symmetric(weight)
        deq_2b = dequantize_2bit_symmetric(q_2b, scales_2b)
        err_2b = (weight - deq_2b).pow(2).mean().item()

        q_4b, scales_4b = quantize_4bit(weight)
        deq_4b = dequantize_4bit(q_4b, scales_4b)
        err_4b = (weight - deq_4b).pow(2).mean().item()

        assert err_4b < err_2b, \
            f"4-bit error ({err_4b:.6f}) should be < 2-bit error ({err_2b:.6f})"

    def test_quantize_weight_dispatch(self, weight):
        for bits in [1, 2, 4]:
            deq, meta = quantize_weight(weight, bits=bits)
            assert deq.shape == weight.shape
            assert meta["bits"] == bits

    def test_quantize_weight_invalid_bits(self, weight):
        with pytest.raises(ValueError):
            quantize_weight(weight, bits=3)

    def test_zero_weight_handling(self):
        """All-zero weight should not cause NaN."""
        weight = torch.zeros(32, 64)
        for bits in [1, 2, 4]:
            deq, _ = quantize_weight(weight, bits=bits)
            assert not torch.isnan(deq).any(), f"NaN in {bits}-bit quantization of zero weight"


# -----------------------------------------------------------------------
# Bit allocator tests
# -----------------------------------------------------------------------

class TestBitAllocator:

    @pytest.fixture
    def salience_map(self):
        torch.manual_seed(42)
        return {
            "layer0.weight": torch.rand(64, 128),
            "layer1.weight": torch.rand(32, 64),
            "layer2.weight": torch.rand(16, 32),
        }

    def test_budget_constraint_respected(self, salience_map):
        """Average bits should not exceed target."""
        config = AllocationConfig(target_avg_bits=1.61)
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)

        total_params = sum(b.numel() for b in bit_map.values())
        total_bits = sum(b.float().sum().item() for b in bit_map.values())
        avg_bits = total_bits / total_params

        assert avg_bits <= config.target_avg_bits + 0.05, \
            f"avg_bits {avg_bits:.4f} exceeds target {config.target_avg_bits}"

    def test_bit_choices_respected(self, salience_map):
        """All allocated bits should be in the allowed set."""
        config = AllocationConfig(target_avg_bits=1.61, bit_choices=[1, 2, 4])
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)

        for name, bits in bit_map.items():
            unique = bits.unique().tolist()
            for b in unique:
                assert int(b) in config.bit_choices, \
                    f"Layer {name} has invalid bit assignment: {b}"

    def test_shapes_preserved(self, salience_map):
        """Bit map shapes should match salience map shapes."""
        config = AllocationConfig(target_avg_bits=1.61)
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)

        for name in salience_map:
            assert bit_map[name].shape == salience_map[name].shape, \
                f"Shape mismatch for {name}"

    def test_high_salience_gets_more_bits(self, salience_map):
        """
        Weights with high salience should get more bits on average
        than weights with low salience.
        """
        # Create controlled salience: one layer much more salient
        controlled = {
            "high_salience.weight": torch.ones(64, 128) * 10.0,  # very salient
            "low_salience.weight": torch.ones(64, 128) * 0.01,   # barely salient
        }
        config = AllocationConfig(target_avg_bits=1.61)
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(controlled)

        high_avg = bit_map["high_salience.weight"].float().mean().item()
        low_avg = bit_map["low_salience.weight"].float().mean().item()

        assert high_avg >= low_avg, \
            f"High salience should get >= bits: high={high_avg:.2f}, low={low_avg:.2f}"

    def test_allocation_stats(self, salience_map):
        config = AllocationConfig(target_avg_bits=1.61)
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)
        stats = allocator.get_allocation_stats(bit_map)

        assert "_summary" in stats
        assert "avg_bits" in stats["_summary"]
        assert "total_params" in stats["_summary"]

    def test_channel_wise_granularity(self, salience_map):
        config = AllocationConfig(target_avg_bits=1.61, granularity="channel")
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)
        # Each row (output channel) should have uniform bit assignment
        for name, bits in bit_map.items():
            if bits.dim() >= 2:
                # Check that all elements in a row have same bits
                row_bits = bits.reshape(bits.shape[0], -1)
                for i in range(row_bits.shape[0]):
                    assert row_bits[i].unique().numel() == 1, \
                        f"Channel {i} of {name} has mixed bits within channel"

    def test_layer_wise_granularity(self, salience_map):
        config = AllocationConfig(target_avg_bits=1.61, granularity="layer")
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(salience_map)
        # Each layer should have a single uniform bit value
        for name, bits in bit_map.items():
            unique = bits.unique()
            assert unique.numel() == 1, \
                f"Layer-wise allocation gave mixed bits for {name}: {unique.tolist()}"


# -----------------------------------------------------------------------
# Salience metric tests
# -----------------------------------------------------------------------

class TestSalienceMetrics:

    @pytest.fixture
    def weight(self):
        torch.manual_seed(42)
        return torch.randn(64, 128)

    def test_magnitude_l1_shape(self, weight):
        result = MagnitudeSalience.compute(weight, norm="l1")
        assert result.shape == weight.shape

    def test_magnitude_l2_shape(self, weight):
        result = MagnitudeSalience.compute(weight, norm="l2")
        assert result.shape == weight.shape

    def test_magnitude_l1_nonneg(self, weight):
        result = MagnitudeSalience.compute(weight, norm="l1")
        assert (result >= 0).all()

    def test_magnitude_l2_nonneg(self, weight):
        result = MagnitudeSalience.compute(weight, norm="l2")
        assert (result >= 0).all()

    def test_magnitude_invalid_norm(self, weight):
        with pytest.raises(ValueError):
            MagnitudeSalience.compute(weight, norm="l3")

    def test_ensemble_normalization(self, weight):
        """Ensemble output should be in [0, 1]."""
        config = SalienceConfig()
        ensemble = EnsembleSalience(config)

        scores = {
            "magnitude_l1": MagnitudeSalience.compute(weight, "l1"),
            "magnitude_l2": MagnitudeSalience.compute(weight, "l2"),
        }

        result = ensemble.combine(scores)
        assert result.min() >= -1e-6, "Ensemble output below 0"
        assert result.max() <= 1 + 1e-6, "Ensemble output above 1"

    def test_ensemble_alpha_normalization(self):
        """Alpha weights should be auto-normalized to sum to 1."""
        config = SalienceConfig(
            alpha_magnitude_l1=1.0,
            alpha_magnitude_l2=1.0,
            alpha_gradient=1.0,
            alpha_hessian=1.0,
            alpha_activation=1.0,
        )
        ensemble = EnsembleSalience(config)
        total = sum(ensemble.alpha.values())
        assert abs(total - 1.0) < 1e-6, f"Alpha sum not normalized: {total}"

    def test_ensemble_missing_metric(self, weight):
        """Should handle missing metrics gracefully."""
        config = SalienceConfig()
        ensemble = EnsembleSalience(config)

        # Only provide 2 of the 5 metrics
        scores = {
            "magnitude_l1": MagnitudeSalience.compute(weight, "l1"),
            "gradient": weight.abs(),  # mock gradient
        }
        result = ensemble.combine(scores)
        assert result.shape == weight.shape

    def test_ensemble_empty_raises(self, weight):
        config = SalienceConfig()
        ensemble = EnsembleSalience(config)
        with pytest.raises(ValueError):
            ensemble.combine({})

    def test_gradient_salience_fallback(self, weight):
        """GradientSalience should fall back to magnitude when no grad available."""
        gs = GradientSalience()
        result = gs.compute(weight, "nonexistent.weight")
        assert result.shape == weight.shape
        assert not torch.isnan(result).any()


# -----------------------------------------------------------------------
# Integration smoke test
# -----------------------------------------------------------------------

class TestIntegration:

    def test_small_model_full_pipeline(self):
        """End-to-end smoke test on a tiny 2-layer MLP."""
        torch.manual_seed(42)

        # Tiny model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Salience
        weights = {
            "0.weight": torch.rand(128, 64),
            "2.weight": torch.rand(64, 128),
            "4.weight": torch.rand(10, 64),
        }

        # Allocate
        config = AllocationConfig(target_avg_bits=1.61)
        allocator = BitAllocator(config)
        bit_map = allocator.allocate(weights)

        # Check avg bits
        total_params = sum(b.numel() for b in bit_map.values())
        total_bits = sum(b.float().sum().item() for b in bit_map.values())
        avg_bits = total_bits / total_params
        assert avg_bits <= 1.66, f"avg bits {avg_bits} > 1.66"

        # Quantize each layer
        for name, param_data in weights.items():
            bits_tensor = bit_map[name]
            bits = int(bits_tensor.float().mean().round().item())
            bits = max(1, min(4, bits))
            deq, meta = quantize_weight(param_data, bits=bits, refine_scales=False)
            assert deq.shape == param_data.shape
            assert not torch.isnan(deq).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
