# Decoder with skips

Assume the encoder skip list was captured as:
`[input_state, conv1, conv2, conv3, conv4, conv5]`.

Forward with skips:

```text
x = ReLU(Linear(512, 2048)(z))
x = view(x, (B, 256, 4, 2))
x = ReLU(ConvT(256,128,3,1,1)(x + encode_list[-1]))
x = bilinear_upsample_2x(x)
x = ReLU(ConvT(128,64,3,1,1)(x + encode_list[-2]))
x = bilinear_upsample_2x(x)
x = ReLU(ConvT(64,32,3,1,1)(x + encode_list[-3]))
x = bilinear_upsample_2x(x)
x = ReLU(ConvT(32,32,5,1,2)(x + encode_list[-4]))
x = bilinear_upsample_2x(x)
x = ConvT(32,5,9,1,4)(x + encode_list[-5])
x = Conv2d(5,128,1,1)(x)
x = ReLU(x)
x = Conv2d(128,5,1,1)(x)
return x
```

The no-skip path exists for completeness, but the assimilation trajectory should use the skip path.
