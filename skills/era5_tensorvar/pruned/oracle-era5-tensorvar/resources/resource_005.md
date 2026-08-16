# State encoder details

Settings:

```text
state_dim = [5, 64, 32]
state_filter_feature_dim = [32, 64, 128, 256]
state_feature_dim = [2048, 512]
```

Forward with skip capture:

```text
encode_list = [state.clone()]
x = Conv2d(5, 32, 9, 1, 4)(state);       encode_list.append(x.clone()); x = ReLU(AvgPool2d(2,2)(x))
x = Conv2d(32, 32, 5, 1, 2)(x);          encode_list.append(x.clone()); x = AvgPool2d(2,2)(ReLU(x))
x = Conv2d(32, 64, 3, 1, 1)(x);          encode_list.append(x.clone()); x = ReLU(AvgPool2d(2,2)(x))
x = Conv2d(64, 128, 3, 1, 1)(x);         encode_list.append(x.clone()); x = ReLU(AvgPool2d(2,2)(x))
x = Conv2d(128, 256, 3, 1, 1)(x);        encode_list.append(x.clone()); x = ReLU(x)
z = Linear(2048, 512)(Flatten(x))
```

The asymmetry in ReLU/pooling order for the second convolution is intentional: it is `pooling(relu(x))`, while several other stages are `relu(pooling(x))`.
