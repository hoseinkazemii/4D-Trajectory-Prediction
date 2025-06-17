import numpy as np

def _aggregate_sequence_predictions(y, row_counts, test_mode, **params):
    sequence_length = params.get("sequence_length")
    prediction_horizon = params.get("prediction_horizon")
    sequence_step = params.get("sequence_step")
    test_indices = params.get("test_indices")
    test_stride_mode = params.get("test_stride_mode")

    total_window = sequence_length + prediction_horizon
    stride = prediction_horizon if test_stride_mode == "prediction_horizon" else total_window

    test_row_counts = [row_counts[i] for i in test_indices]
    scenario_nseqs = []

    for c in test_row_counts:
        diff = c - total_window
        if test_mode:
            n_seq = (diff // stride) + 1 if diff >= 0 else 0
        else:
            n_seq = (diff // sequence_step) + 1 if diff >= 0 else 0
        scenario_nseqs.append(max(n_seq, 0))

    def _aggregate_partial_overlap(y_chunk):
        n_seq, horizon, n_feat = y_chunk.shape
        final_len = n_seq + horizon - 1
        sums = np.zeros((final_len, n_feat))
        counts = np.zeros(final_len, dtype=np.float32)

        for i in range(n_seq):
            for p in range(horizon):
                t = i + p
                sums[t] += y_chunk[i, p]
                counts[t] += 1

        return np.divide(sums, counts[:, None], where=counts[:, None] != 0)

    def _aggregate_nonoverlap(y_chunk):
        n_seq, horizon, n_feat = y_chunk.shape
        aggregator = np.zeros((n_seq * horizon, n_feat), dtype=np.float32)
        for i in range(n_seq):
            aggregator[i*horizon:(i+1)*horizon] = y_chunk[i]
        return aggregator

    aggregated_list = []
    idx_start = 0

    for i, n_seq in enumerate(scenario_nseqs):
        if n_seq <= 0:
            continue
        idx_end = idx_start + n_seq
        y_chunk = y[idx_start: idx_end]
        idx_start = idx_end

        scen_agg = _aggregate_nonoverlap(y_chunk) if test_mode else _aggregate_partial_overlap(y_chunk)
        aggregated_list.append(scen_agg)

    return np.concatenate(aggregated_list, axis=0) if aggregated_list else np.array([])
