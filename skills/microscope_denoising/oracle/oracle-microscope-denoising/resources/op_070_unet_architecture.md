# U-Net

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(base*8, base*8)
        self.dec4 = DoubleConv(base*8 + base*8, base*4)
        self.dec3 = DoubleConv(base*4 + base*4, base*2)
        self.dec2 = DoubleConv(base*2 + base*2, base)
        self.dec1 = DoubleConv(base + base, base)
        self.out_conv = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        def up(z, s): return F.interpolate(z, size=s.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([up(b, e4), e4], 1))
        d3 = self.dec3(torch.cat([up(d4, e3), e3], 1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        return self.out_conv(d1)
```
