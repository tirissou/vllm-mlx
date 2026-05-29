import mlx.core as mx
from vllm_mlx.cache_types import StaticKVData, StaticRecurrentData  # canonical types


class CacheTranslator:
    @staticmethod
    def linearize(tensor: mx.array, offset: int, max_size: int) -> mx.array:
        """Transforms a circular buffer into a contiguous, linear view."""
        if offset == max_size:
            return tensor[..., :offset, :]
        part1 = tensor[..., offset:, :]
        part2 = tensor[..., :offset, :]
        return mx.concatenate([part1, part2], axis=-2)

    @staticmethod
    def quantize_kv(kv_arrays: list[mx.array]) -> tuple[list[mx.array], list[float]]:
        """Performs per-tensor int8 quantization."""
        quantized: list[mx.array] = []
        scales: list[float] = []
        for arr in kv_arrays:
            arr_f32 = arr.astype(mx.float32)
            max_val = mx.max(mx.abs(arr_f32)).item()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q = mx.clip(mx.round(arr_f32 / scale), -127, 127).astype(mx.int8)
            quantized.append(q)
            scales.append(scale)
        return quantized, scales

    @staticmethod
    def dequantize_kv(quantized_arrays: list[mx.array], scales: list[float]) -> list[mx.array]:
        """Performs dequantization math."""
        return [
            (arr.astype(mx.float32) * scale).astype(mx.bfloat16)
            for arr, scale in zip(quantized_arrays, scales)
        ]
