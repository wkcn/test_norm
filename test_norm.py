# reference: https://zhuanlan.zhihu.com/p/395855181
import torch
import torch.nn.functional as F
import numpy as np

N = 8
C = 31
H = 7
W = 14
L = H * W
eps = 1e-5

# Batch Norm
x = torch.randn((N, C, H, W))
axes = (0, 2, 3)
var = x.var(axes, unbiased=False, keepdim=True)
mean = x.mean(axes, keepdim=True)
bn_out = F.batch_norm(x, None, None, training=True, eps=eps)
bn_out2 = (x - mean) / (var + eps).sqrt()

# print(bn_out.flatten()[:10], bn_out2.flatten()[:10])
np.testing.assert_almost_equal(bn_out.data.numpy(), bn_out2.data.numpy(), decimal=5)

# Layer Norm
x = torch.randn((N, L, C))
axes = (-1, )
var = x.var(axes, unbiased=False, keepdim=True)
mean = x.mean(axes, keepdim=True)
ln_out = F.layer_norm(x, normalized_shape=(C, ), eps=eps)
ln_out2 = (x - mean) / (var + eps).sqrt()

np.testing.assert_almost_equal(ln_out.data.numpy(), ln_out2.data.numpy(), decimal=5)

x = torch.randn((N, C, H, W))
axes = (1, 2, 3)
var = x.var(axes, unbiased=False, keepdim=True)
mean = x.mean(axes, keepdim=True)
ln_out = F.layer_norm(x, normalized_shape=(C, H, W), eps=eps)
ln_out2 = (x - mean) / (var + eps).sqrt()

np.testing.assert_almost_equal(ln_out.data.numpy(), ln_out2.data.numpy(), decimal=5)

# Instance Norm
x = torch.randn((N, C, H, W))
axes = (2, 3)
var = x.var(axes, unbiased=False, keepdim=True)
mean = x.mean(axes, keepdim=True)
in_out = F.instance_norm(x, eps=eps)
in_out2 = (x - mean) / (var + eps).sqrt()

np.testing.assert_almost_equal(in_out.data.numpy(), in_out2.data.numpy(), decimal=5)
