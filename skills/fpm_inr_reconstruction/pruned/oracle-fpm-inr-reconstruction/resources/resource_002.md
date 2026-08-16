# INR architecture — reference excerpt

```python
import torch, torch.nn as nn

class G_Renderer(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False):
        super().__init__()
        act = nn.ReLU(); layers = [nn.Linear(in_dim, hidden_dim)]
        if use_layernorm: layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act)
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm: layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class G_Tensor3D(nn.Module):
    def __init__(self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(2e-4 * torch.randn((x_mode, y_mode, num_feats)))
        self.renderer = G_Renderer(in_dim=num_feats, use_layernorm=use_layernorm)
        self.x0 = None
        self.z_mode = z_dim
        self.z_data = nn.Parameter(torch.randn((z_dim, num_feats)))
        self.z_min, self.z_max, self.z_dim = z_min, z_max, z_dim
    def create_coords(self, x_dim, y_dim, x_max, y_max):
        hdx, hdy = 0.5/x_dim, 0.5/y_dim
        xs = torch.linspace(hdx, 1-hdx, x_dim); ys = torch.linspace(hdx, 1-hdy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing='ij')
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max]).float(); idx = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(xy[None], requires_grad=False)
        self.x0 = nn.Parameter(idx[:,0].clamp(0, x_max-1), requires_grad=False)
        self.y0 = nn.Parameter(idx[:,1].clamp(0, y_max-1), requires_grad=False)
        self.x1 = nn.Parameter((self.x0+1).clamp(max=x_max-1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0+1).clamp(max=y_max-1), requires_grad=False)
        self.lerp_weights = nn.Parameter(xs - idx.float(), requires_grad=False)
    def normalize_z(self, z): return (self.z_dim-1)*(z - self.z_min)/(self.z_max - self.z_min)
    def sample(self, z):
        z = self.normalize_z(z); z0 = z.long().clamp(0, self.z_dim-1); z1 = (z0+1).clamp(max=self.z_dim-1)
        zw = (z - z.long().float())[:, None]
        lw = self.lerp_weights
        xy_feat = (self.data[self.y0,self.x0]*(1-lw[:,0:1])*(1-lw[:,1:2])
                 + self.data[self.y0,self.x1]*lw[:,0:1]*(1-lw[:,1:2])
                 + self.data[self.y1,self.x0]*(1-lw[:,0:1])*lw[:,1:2]
                 + self.data[self.y1,self.x1]*lw[:,0:1]*lw[:,1:2])
        z_feat = self.z_data[z0]*(1-zw) + self.z_data[z1]*zw
        z_feat = z_feat[:,None].repeat(1, xy_feat.shape[0], 1)
        return xy_feat[None].repeat(z.shape[0],1,1) * z_feat
    def forward(self, z):
        out = self.renderer(self.sample(z))
        return out.view(z.shape[0], 1, self.x_dim, self.y_dim)

class FullModel(nn.Module):
    def __init__(self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm):
        super().__init__()
        self.img_real = G_Tensor3D(x_mode,y_mode,5,z_min,z_max,num_feats,use_layernorm)
        self.img_imag = G_Tensor3D(x_mode,y_mode,5,z_min,z_max,num_feats,use_layernorm)
        self.w, self.h = w, h; self.init_scale_grids(ds_factor)
    def init_scale_grids(self, ds_factor):
        self.img_real.create_coords(self.w//ds_factor, self.h//ds_factor, self.img_real.x_mode, self.img_real.y_mode)
        self.img_imag.create_coords(self.w//ds_factor, self.h//ds_factor, self.img_imag.x_mode, self.img_imag.y_mode)
        self.ds_factor = ds_factor
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode='bilinear')
    def forward(self, dz):
        r = self.us_module(self.img_real(dz)).squeeze(1)
        i = self.us_module(self.img_imag(dz)).squeeze(1)
        return r, i
```

Notes: `z_dim` is fixed to 5 inside FullModel regardless of any config. Spatial init uses `2e-4*randn`; depth init plain `randn`. The MLP is intentionally tiny (2 hidden ReLU layers, width 32).
