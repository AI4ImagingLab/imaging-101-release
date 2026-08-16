# Batch and array conventions

`raw_data.npz` keys:

| key | stored shape | selected shape | role |
| --- | --- | --- | --- |
| `obs_history` | `(1, 5, 50, 64, 32)` | `(5, 50, 64, 32)` | normalized observation histories |
| `max_val` | `(1, 5)` | `(5,)` | channel maxima for optional diagnostics only |
| `min_val` | `(1, 5)` | `(5,)` | channel minima for optional diagnostics only |
| `lat_weight_matrix` | `(1, 5, 64, 32)` | `(5, 64, 32)` | cosine-latitude metric weights |

Sample selection is a direct index operation: `{k: v[0] for k, v in arrays.items()}`. Do not squeeze arbitrary axes; only remove the leading sample axis.
