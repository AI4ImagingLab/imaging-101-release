# BH-NeRF MLP pseudocode

```
in_features = 3 + 2*3*posenc_deg   # 21 for deg=3
skip_layer  = net_depth // 2       # 2

layers[i] dims:
  i == 0:                in_features -> net_width
  i == skip_layer (>0):  net_width + in_features -> net_width
  else:                  net_width -> net_width
output_layer: net_width -> 1

init: kaiming_uniform_(weight, nonlinearity='relu'); zeros_(bias)

forward(encoded):
  x = encoded
  for i, layer in enumerate(layers):
    if i == skip_layer and i > 0:
      x = concat([x, encoded], dim=-1)
    x = relu(layer(x))
  return output_layer(x)          # logits, shape (*, 1)

emission = sigmoid(logits[...,0] - 10.0)
```

The `-10.0` offset is essential: it initializes emission near 0 everywhere.
