import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ────────────────────────── 1. CUDA core implementation (based on HA4DETR) ──────────────────────────
CPP_STUB = r"""
#include <torch/extension.h>
void hungarian_launcher(torch::Tensor cost, torch::Tensor ncols, torch::Tensor assignment);
"""

CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define _MAX_ROWS 1369
#define _MAX_COLS 1369
#define _BLOCK_SIZE 512
#define _WARP 32

constexpr int MAX_ROWS   = _MAX_ROWS;       
constexpr int MAX_COLS   = _MAX_COLS;       
constexpr int WARP       = _WARP;
constexpr float INF      = 1e20f;

template<int NT>
__device__ __forceinline__ void warp_min_reduce(float &val, int &idx) {
    #pragma unroll
    for (int offset = NT / 2; offset > 0; offset >>= 1) {
        float val_other = __shfl_down_sync(0xffffffff, val , offset);
        int   idx_other = __shfl_down_sync(0xffffffff, idx , offset);
        if (val_other < val) { val = val_other; idx = idx_other; }
    }
}

template<int BLOCK_SIZE>
__global__ void hungarian_kernel(const float* cost, const int* ncols, int* assignment, int B, int cost_stride, int asgn_stride) {
    int globalBid = blockIdx.x;
    const int tid = threadIdx.x;

    while (globalBid < B) {
        const float *costB = cost + globalBid * cost_stride;
        const int    cols  = ncols[globalBid];
        int         *asgnB = assignment + globalBid * asgn_stride;

        __shared__ float u[MAX_COLS + 1];
        __shared__ float v[MAX_ROWS + 1];
        __shared__ int   p[MAX_ROWS + 1];
        __shared__ int   way[MAX_ROWS + 1];
        __shared__ float minv[MAX_ROWS + 1];
        __shared__ bool  used[MAX_ROWS + 1];
        __shared__ int   j0;
        __shared__ float delta_s;
        __shared__ int   j1_s;
        __shared__ bool  path_found;

        for (int k = tid; k <= MAX_ROWS; k += BLOCK_SIZE) { v[k] = u[k] = 0.0f; p[k] = 0; }
        __syncthreads();

        for (int i = 1; i <= cols; ++i) {
            if (tid == 0) { p[0] = i; j0 = 0; path_found = false; }
            __syncthreads();
            for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) { minv[j] = INF; used[j] = false; }
            __syncthreads();

            while (!path_found) {
                if (tid == 0) used[j0] = true;
                __syncthreads();
                int i0  = p[j0];
                float best_val = INF; int best_j = 0;
                for (int j = tid + 1; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) continue;
                    float cur = costB[(j - 1) * MAX_COLS + (i0 - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < best_val) { best_val = minv[j]; best_j = j; }
                }
                const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP;
                warp_min_reduce<WARP>(best_val, best_j);
                if (BLOCK_SIZE > WARP) {
                    __shared__ float warp_min_val[WARPS_PER_BLOCK];
                    __shared__ int   warp_min_j[WARPS_PER_BLOCK];
                    if ((tid & (WARP - 1)) == 0) { warp_min_val[tid / WARP] = best_val; warp_min_j[tid / WARP] = best_j; }
                    __syncthreads();
                    if (tid < WARPS_PER_BLOCK) { best_val = warp_min_val[tid]; best_j = warp_min_j[tid]; }
                    else best_val = INF;
                    if (tid == 0) {
                        for (int k = 1; k < WARPS_PER_BLOCK; ++k)
                            if (warp_min_val[k] < best_val) { best_val = warp_min_val[k]; best_j = warp_min_j[k]; }
                        delta_s = best_val; j1_s = best_j;
                    }
                } else if (tid == 0) { delta_s = best_val; j1_s = best_j; }
                __syncthreads();

                for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) { u[p[j]] += delta_s; v[j] -= delta_s; } else minv[j] -= delta_s;
                }
                __syncthreads();
                if (tid == 0) { j0 = j1_s; if (p[j0] == 0) path_found = true; }
                __syncthreads();
            }
            if (tid == 0) { while (j0 != 0) { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } }
            __syncthreads();
        }
        for (int row = tid + 1; row <= MAX_ROWS; row += BLOCK_SIZE) {
            int task = p[row];
            asgnB[(row - 1) * 2] = row - 1;
            asgnB[(row - 1) * 2 + 1] = task - 1;
        }
        globalBid += gridDim.x;
        __syncthreads();
    }
}

void hungarian_launcher(torch::Tensor cost, torch::Tensor ncols, torch::Tensor assignment) {
    const int B = cost.size(0);
    const int R = cost.size(1);
    const int C = cost.size(2);
    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    int blocks  = std::min(B, smCount * 4);
    hungarian_kernel<_BLOCK_SIZE><<<blocks, _BLOCK_SIZE>>>(
        cost.data_ptr<float>(), ncols.data_ptr<int>(), assignment.data_ptr<int>(), B, R * C, C * 2);
}
"""

# Compile and load the CUDA extension
ext_mod = load_inline(
    name="hungarian_gpu_lib",
    cpp_sources=[CPP_STUB],
    cuda_sources=[CUDA_SRC],
    functions=["hungarian_launcher"],
    verbose=False,
)


def hungarian_gpu(cost: torch.Tensor, ncols: torch.Tensor) -> torch.Tensor:
    B, _, C = cost.shape
    out = torch.empty((B, C, 2), dtype=torch.int32, device=cost.device)
    ext_mod.hungarian_launcher(cost, ncols, out)
    return out


# ────────────────────────── 2. Wrapped loss computation class ──────────────────────────

class HungarianMatcherWithLossGPU(nn.Module):
    def __init__(self, cost_cosine: float = 1.0):
        super().__init__()
        self.cost_cosine = cost_cosine
        # Constant limits matching the CUDA code
        self._MAX_ROWS = 1369
        self._MAX_COLS = 1369

    def forward(self, features1, features2):
        """
        features1: [batch, T1, channel]
        features2: [batch, T2, channel]
        """
        B, T1, C = features1.shape
        T2 = features2.shape[1]
        device = features1.device

        # 1. Compute the cosine similarity cost [B, T1, T2]
        f1_n = F.normalize(features1, p=2, dim=-1)
        f2_n = F.normalize(features2, p=2, dim=-1)
        sim = torch.bmm(f1_n, f2_n.transpose(1, 2))
        cost = self.cost_cosine * (1.0 - sim)

        # 2. Build a fixed-size padded matrix as required by CUDA
        # Initialize the cost to the very large value INF
        padded_cost = torch.full((B, self._MAX_ROWS, self._MAX_COLS), 1e20,
                                 device=device, dtype=torch.float32)
        padded_cost[:, :T1, :T2] = cost

        # Record the actual number of columns
        Ns = torch.full((B,), T2, dtype=torch.int32, device=device)

        # 3. Invoke the efficient GPU matching
        # output shape [B, _MAX_COLS, 2], where the 2nd column is the matched features2 index
        with torch.no_grad():
            output = hungarian_gpu(padded_cost, Ns)
            # We only need the T1 rows corresponding to features1
            # output[:, :, 1] stores, for each row of features1, the column index of the matched features2
            matched_indices = output[:, :T1, 1].long()

        # 4. Compute the average loss over matched pairs
        # Extract similarity values from the original sim matrix via indexing
        matched_sim = torch.gather(sim, dim=2, index=matched_indices.unsqueeze(-1)).squeeze(-1)

        # Loss = 1 - Mean(CosineSimilarity)
        loss = 1.0 - matched_sim.mean()

        return loss