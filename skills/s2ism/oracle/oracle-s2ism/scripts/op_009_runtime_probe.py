#!/usr/bin/env python
"""Cheap runtime probe for the s2ISM public environment."""
import importlib.util
import sys
print('python:', sys.version.split()[0])
for mod in ['numpy', 'torch']:
    spec = importlib.util.find_spec(mod)
    print(mod, 'available' if spec else 'missing')
if importlib.util.find_spec('torch'):
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
