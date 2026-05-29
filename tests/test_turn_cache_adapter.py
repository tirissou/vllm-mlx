import mlx.core as mx
import pytest
from vllm_mlx.turn_cache_adapter import TurnCacheAdapter
from vllm_mlx.cache_types import StaticKVData, StaticRecurrentData


def test_segment_rotating_sets_layer_index_and_merge_strategy():
    adapter = TurnCacheAdapter()
    raw_keys = mx.array([[[[1.0], [2.0]]]], dtype=mx.float32)
    raw_values = mx.array([[[[3.0], [4.0]]]], dtype=mx.float32)
    live_states = [{
        'class_name': 'RotatingKVCache',
        'state': (raw_keys, raw_values),
        'meta_state': (0, 2, 1, 2),  # keep, max_size, offset, _
    }]

    kv_data, rec_data = adapter.segment(live_states)

    assert kv_data[0] is not None
    assert rec_data[0] is None
    assert kv_data[0].metadata['layer_index'] == 0
    assert kv_data[0].metadata['merge_strategy'] == 'last'
    assert kv_data[0].metadata['class_name'] == 'RotatingKVCache'


def test_segment_kvcache_sets_layer_index_and_merge_strategy():
    adapter = TurnCacheAdapter()
    raw_keys = mx.array([[[[1.0], [2.0], [3.0]]]], dtype=mx.float32)
    raw_values = mx.array([[[[4.0], [5.0], [6.0]]]], dtype=mx.float32)
    live_states = [{
        'class_name': 'KVCache',
        'state': (raw_keys, raw_values),
        'meta_state': (2,),
    }]

    kv_data, rec_data = adapter.segment(live_states)

    assert kv_data[0] is not None
    assert kv_data[0].metadata['layer_index'] == 0
    assert kv_data[0].metadata['merge_strategy'] == 'concatenate'


def test_segment_recurrent_sets_layer_index():
    adapter = TurnCacheAdapter()
    rec_state = [{'state': (mx.array([[[[1.0]]]], dtype=mx.bfloat16),), 'class_name': 'MambaLayer'}]
    live_states = [{'class_name': 'MambaLayer', 'state': rec_state, 'meta_state': ()}]

    kv_data, rec_data = adapter.segment(live_states)

    assert kv_data[0] is None
    assert rec_data[0] is not None
    assert rec_data[0].metadata['layer_index'] == 0
    assert rec_data[0].metadata['class_name'] == 'MambaLayer'


def test_round_trip_kvcache():
    adapter = TurnCacheAdapter()
    raw_keys = mx.array([[[[1.0], [2.0], [3.0]]]], dtype=mx.float32)
    raw_values = mx.array([[[[4.0], [5.0], [6.0]]]], dtype=mx.float32)
    live_states = [{'class_name': 'KVCache', 'state': (raw_keys, raw_values), 'meta_state': (2,)}]

    kv_sparse, rec_sparse = adapter.segment(live_states)
    kv_compact = [kv for kv in kv_sparse if kv is not None]
    rec_compact = [rec for rec in rec_sparse if rec is not None]

    result = adapter.assemble(kv_compact, rec_compact)

    assert len(result) == 1
    assert result[0]['class_name'] == 'KVCache'
    keys, values = result[0]['state']
    assert keys.shape == (1, 1, 2, 1)
    assert values.shape == (1, 1, 2, 1)


def test_round_trip_rotating():
    adapter = TurnCacheAdapter()
    raw_keys = mx.array([[[[1.0], [2.0]]]], dtype=mx.float32)
    raw_values = mx.array([[[[3.0], [4.0]]]], dtype=mx.float32)
    live_states = [{'class_name': 'RotatingKVCache', 'state': (raw_keys, raw_values), 'meta_state': (0, 2, 1, 2)}]

    kv_sparse, rec_sparse = adapter.segment(live_states)
    kv_compact = [kv for kv in kv_sparse if kv is not None]
    rec_compact = [rec for rec in rec_sparse if rec is not None]

    result = adapter.assemble(kv_compact, rec_compact)

    assert len(result) == 1
    assert result[0]['class_name'] == 'RotatingKVCache'


def test_assemble_multi_layer_ordering():
    """assemble() must return layers sorted by layer_index regardless of input order."""
    adapter = TurnCacheAdapter()
    # Simulate retrieval where kv layers arrive out of order
    kv2 = StaticKVData(
        arrays=[mx.zeros((1, 1, 2, 1), dtype=mx.int8), mx.zeros((1, 1, 2, 1), dtype=mx.int8)],
        metadata={'class_name': 'KVCache', 'layer_index': 2, 'merge_strategy': 'concatenate',
                  'actual_end': 2, 'scales': [1.0, 1.0]},
    )
    kv0 = StaticKVData(
        arrays=[mx.zeros((1, 1, 2, 1), dtype=mx.int8), mx.zeros((1, 1, 2, 1), dtype=mx.int8)],
        metadata={'class_name': 'KVCache', 'layer_index': 0, 'merge_strategy': 'concatenate',
                  'actual_end': 2, 'scales': [1.0, 1.0]},
    )
    result = adapter.assemble([kv2, kv0], [])
    assert result[0]['class_name'] == 'KVCache'
    # layer_index 0 must come first regardless of input order
    result_a = adapter.assemble([kv2, kv0], [])
    result_b = adapter.assemble([kv0, kv2], [])
    assert result_a[0]['class_name'] == result_b[0]['class_name']
    assert result_a[0]['meta_state'] == result_b[0]['meta_state']
