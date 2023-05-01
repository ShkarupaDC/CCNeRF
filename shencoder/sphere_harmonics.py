import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _shencoder as _backend
except ImportError:
    from .backend import _backend

class _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape # batch size, coord dim
        output_dim = degree ** 2

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = torch.empty(1, dtype=inputs.dtype, device=inputs.device)

        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, calc_grad_inputs, dy_dx)

        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        if ctx.calc_grad_inputs:
            grad = grad.contiguous()
            inputs, dy_dx = ctx.saved_tensors
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None



sh_encode = _sh_encoder.apply

# NOTE(dsh): shencoder.cu rewritten in pure PyTorch
def components_from_spherical_harmonics(directions: torch.Tensor, levels: int) -> torch.Tensor:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels**2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = -0.48860251190291987 * y
        components[..., 2] = 0.48860251190291987 * z
        components[..., 3] = -0.48860251190291987 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = -1.0925484305920792 * y * z
        components[..., 6] = 0.94617469575755997 * zz - 0.31539156525251999
        components[..., 7] = -1.0925484305920792 * x * z
        components[..., 8] = 0.54627421529603959 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.59004358992664352 * y * (-3 * xx + yy)
        components[..., 10] = 2.8906114426405538 * x * y * z
        components[..., 11] = 0.45704579946446572 * y * (1 - 5 * zz)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.45704579946446572 * x * (1 - 5 * zz)
        components[..., 14] = 1.4453057213202769 * z * (xx - yy)
        components[..., 15] = 0.59004358992664352 * x * (-xx + 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (-3 * xx + yy)
        components[..., 18] = 0.94617469575756008 * x * y * (7 * zz - 1)
        components[..., 19] = 0.66904654355728921 * y * (3 - 7 * zz)
        components[..., 20] = -3.1735664074561294 * zz + 3.7024941420321507 * zz * zz + 0.31735664074561293
        components[..., 21] = 0.66904654355728921 * x * z * (3 - 7 * zz)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (-xx + 3 * yy)
        components[..., 24] = -3.7550144126950569 * xx * yy + 0.62583573544917614 * xx * xx + 0.62583573544917614 * yy * yy

    return components


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4, use_cuda_version=False):
        super().__init__()

        self.input_dim = input_dim # coord dims, must be 3
        self.degree = degree # 0 ~ 4
        self.output_dim = degree ** 2
        self.use_cuda_version = use_cuda_version

        assert self.input_dim == 3, "SH encoder only support input dim == 3"
        assert self.degree > 0 and self.degree <= 8, "SH encoder only supports degree in [1, 8]"

    def __repr__(self):
        return f"SHEncoder: input_dim={self.input_dim} degree={self.degree}"

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., degree^2]

        inputs = inputs / size # [-1, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)

        if self.use_cuda_version:
            outputs = sh_encode(inputs, self.degree, inputs.requires_grad)
        else:
            # with torch.no_grad():
            outputs = components_from_spherical_harmonics(inputs, self.degree)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs
