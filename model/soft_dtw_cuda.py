# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

@cuda.jit
def compute_softdtw_cuda(D, gamma, warp, bandwidth, max_i, max_j, n_passes, R):

    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid

    inv_gamma = 1.0 / gamma

    for p in range(n_passes):

        J = max(0, min(p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                r0 = -(R[b, i - 1, j - 1] + D[b, i - 1, j - 1]) * inv_gamma
                r1 = -(R[b, i - 1, j] + D[b, i - 1, j] + warp) * inv_gamma
                r2 = -(R[b, i, j - 1] + D[b, i, j - 1] + warp) * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = softmin

        # Wait for other threads in this block
        cuda.syncthreads()

@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, warp, bandwidth, max_i, max_j, n_passes, E, G):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1

        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i, j] - warp) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j] - warp) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i, j]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                G[k, i, j] = E[k, i + 1, j]+E[k, i, j+1]+E[k, i+1, j+1]

        cuda.syncthreads()

class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, X, raw_D, D, gamma, warp, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        warp = torch.cuda.FloatTensor([warp])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N + 1, M + 1)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        # Prepare the output array
        R = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
        R[:, :, 0] = torch.ones((B, 1), device=dev, dtype=dtype) * math.inf
        R[:, 0, :] = torch.ones((B, 1), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D_.detach()),
                                                   gamma.item(), warp.item(), bandwidth.item(), N + 1, M + 1, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(X, raw_D, D, R, gamma, warp, bandwidth)
        return R[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        X, raw_D, D, R, gamma, warp, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        H = X.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        G = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        G[:, -1, -1] = 1

        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), warp.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E), cuda.as_cuda_array(G))
        G = G[:, 1:N + 1, 1:M + 1] # dR_D

        tmp_G = G.unsqueeze(-1).expand(-1, -1, -1, H)
        tmp_G = tmp_G * torch.sign(raw_D)
        dR_X = tmp_G.sum(dim=2)

        return grad_output.view(-1, 1, 1).expand_as(dR_X) * dR_X, None, None, None, None, None


@jit(nopython=True)
def cpu_compute_softdtw(D, gamma, warp, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    D_ = np.zeros((B, N + 2, M + 2))
    D_[:, 1:N + 1, 1:M + 1] = D
    R = np.zeros((B, N + 2, M + 2))
    R[:, :, 0] = np.ones((B, 1)) * np.inf
    R[:, 0, :] = np.ones((B, 1)) * np.inf
    R[:, 0, 0] = 0
    for b in range(B):
        for j in range(1, M + 2):
            for i in range(1, N + 2):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -(R[b, i - 1, j - 1] + D_[b, i - 1, j - 1]) / gamma
                r1 = -(R[b, i - 1, j] + D_[b, i - 1, j] + warp) / gamma
                r2 = -(R[b, i, j - 1] + D_[b, i, j - 1] + warp) / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def cpu_compute_softdtw_backward(D_, R, gamma, warp, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    G = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    G[:, -1, -1] = 1
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i, j] - warp) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j] - warp) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i, j]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                G[k, i, j] = E[k, i + 1, j]+E[k, i, j+1]+E[k, i+1, j+1]

    return G[:, 1:N + 1, 1:M + 1]


class CPUSoftDTW(Function):

    @staticmethod
    def forward(ctx, X, raw_D, D, gamma, warp, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        warp = torch.Tensor([warp]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        w_ = warp.item()
        b_ = bandwidth.item()
        R = torch.Tensor(cpu_compute_softdtw(D_, g_, w_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(X, raw_D, D, R, gamma, warp, bandwidth)
        return R[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        X, raw_D, D, R, gamma, warp, bandwidth = ctx.saved_tensors
        H = X.shape[2]
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        w_ = warp.item()
        b_ = bandwidth.item()
        G = torch.Tensor(cpu_compute_softdtw_backward(D_, R_, g_, w_, b_)).to(dev).type(dtype)
        tmp_G = G.unsqueeze(-1).expand(-1, -1, -1, H)
        tmp_G = tmp_G * torch.sign(raw_D)
        dR_X = tmp_G.sum(dim=2)

        return grad_output.view(-1, 1, 1).expand_as(dR_X) * dR_X, None, None, None, None, None



class SoftDTW(torch.nn.Module):

    def __init__(self, use_cuda, gamma=1.0, warp=1.0, normalize=False, bandwidth=None, dist_func=None):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.warp = warp
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        self.dist_func = SoftDTW._manhattan_dist_func

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        return _SoftDTWCUDA.apply if use_cuda else CPUSoftDTW.apply

    @staticmethod
    def _manhattan_dist_func(x, y):
        """
        Calculates the Manhattan distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.abs(x - y).sum(3), (x - y)

    def forward(self, X, Y):

        func_dtw = self._get_func_dtw(X, Y)

        D_xy, raw_D_xy = self.dist_func(X, Y)
        return func_dtw(X, raw_D_xy, D_xy, self.gamma, self.warp, self.bandwidth)

