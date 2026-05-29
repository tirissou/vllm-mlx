from typing import Any
import mlx.core as mx
from vllm_mlx.cache_types import StaticKVData, StaticRecurrentData
from vllm_mlx.cache_translator import CacheTranslator


class TurnCacheAdapter:
    def segment(
        self, live_states: list[dict[str, Any]]
    ) -> tuple[list[StaticKVData | None], list[StaticRecurrentData | None]]:
        """Transform live cache states into normalized static format.

        Returns two sparse lists parallel to live_states: kv_data_list and rec_data_list.
        Exactly one of kv_data_list[i] / rec_data_list[i] is non-None for each i.
        """
        kv_data_list: list[StaticKVData | None] = [None] * len(live_states)
        rec_data_list: list[StaticRecurrentData | None] = [None] * len(live_states)

        for i, state_dict in enumerate(live_states):
            class_name = state_dict['class_name']
            state = state_dict['state']
            meta = state_dict.get('meta_state', ())

            if class_name == 'RotatingKVCache':
                try:
                    keep, max_size, offset, _ = map(int, meta)
                except (TypeError, ValueError):
                    max_size = state[0].shape[2]
                    offset = max_size
                    keep = 0

                raw_keys, raw_values = state
                lin_keys = CacheTranslator.linearize(raw_keys, offset, max_size)
                lin_values = CacheTranslator.linearize(raw_values, offset, max_size)
                q_arrays, q_scales = CacheTranslator.quantize_kv([lin_keys, lin_values])

                kv_data_list[i] = StaticKVData(
                    arrays=q_arrays,
                    metadata={
                        'class_name': 'RotatingKVCache',
                        'layer_index': i,
                        'merge_strategy': 'last',
                        'max_size': max_size,
                        'keep': keep,
                        'offset': offset,
                        'scales': q_scales,
                    },
                )

            elif 'KVCache' in class_name:
                try:
                    actual_end = int(meta[0]) if meta else state[0].shape[2]
                except (TypeError, ValueError, IndexError):
                    actual_end = state[0].shape[2]

                sliced = [arr[:, :, :actual_end, :] for arr in state[:2]]
                q_arrays, q_scales = CacheTranslator.quantize_kv(sliced)

                kv_data_list[i] = StaticKVData(
                    arrays=q_arrays,
                    metadata={
                        'class_name': class_name,
                        'layer_index': i,
                        'merge_strategy': 'concatenate',
                        'actual_end': actual_end,
                        'scales': q_scales,
                    },
                )

            else:
                rec_data_list[i] = StaticRecurrentData(
                    arrays=state,
                    metadata={'class_name': class_name, 'layer_index': i},
                )

        return kv_data_list, rec_data_list

    def assemble(
        self,
        kv_data: list[StaticKVData],
        recurrent_data: list[StaticRecurrentData],
    ) -> list[dict[str, Any]]:
        """Reconstruct live cache states from compact static data lists.

        Input lists are non-sparse (no None entries). Output is ordered by layer_index.
        """
        all_states: dict[int, dict[str, Any]] = {}

        for kv in kv_data:
            li = kv.metadata['layer_index']
            class_name = kv.metadata['class_name']
            scales = kv.metadata['scales']
            dq = CacheTranslator.dequantize_kv(kv.arrays, scales)

            if class_name == 'RotatingKVCache':
                all_states[li] = {
                    'class_name': class_name,
                    'state': (dq[0], dq[1]),
                    'meta_state': (
                        kv.metadata['max_size'],
                        kv.metadata['keep'],
                        kv.metadata['offset'],
                    ),
                }
            else:
                all_states[li] = {
                    'class_name': class_name,
                    'state': tuple(dq),
                    'meta_state': (kv.metadata.get('actual_end', dq[0].shape[-2]),),
                }

        for rec in recurrent_data:
            li = rec.metadata['layer_index']
            all_states[li] = {
                'class_name': rec.metadata['class_name'],
                'state': rec.arrays,
                'meta_state': (),
            }

        return [all_states[li] for li in sorted(all_states)]
