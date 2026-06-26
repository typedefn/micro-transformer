#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
// ----------------------------------------------------------------------
// CUSTOM HIP KERNELS FOR ROW-MAJOR MATRIX MULTIPLICATION
// ----------------------------------------------------------------------

__global__ void kComputeLogitGradients(const float* predictions, const float* targets, float* gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Logit gradient formula: (Predictions - Targets)
        gradients[idx] = predictions[idx] - targets[idx];
    }
}

__global__ void kSoftCrossEntropy(const float* predicted, const float* targets, float* total_loss, int size, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reduction
    __shared__ float sdata[BLOCK_SIZE];
    float local_loss = 0.0f;

    if (idx < size) {
        float p = predicted[idx];
        // Prevent log(0) and log(negative)
        p = fmaxf(epsilon, fminf(1.0f - epsilon, p));
        local_loss = -targets[idx] * logf(p);
    }

    sdata[threadIdx.x] = local_loss;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(total_loss, sdata[0]);
}


// One thread per element in the final concatenated matrix
__global__ void kConcatenateHeads(float** head_ptrs, float* dest, int num_heads, int head_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total_tokens * num_heads * head_dim;
    
    if (idx < total_elements) {
        int token_idx = idx / (num_heads * head_dim);
        int feature_idx = idx % (num_heads * head_dim);
        int head_idx = feature_idx / head_dim;
        int head_feature_idx = feature_idx % head_dim;
        
        // head_ptrs is an array of device pointers to each head's data
        dest[idx] = head_ptrs[head_idx][token_idx * head_dim + head_feature_idx];
    }
}

// --- LAYER NORM KERNELS ---

// 1. Compute Mean and Variance (Fused Kernel)
// One block per row (sequence element). Threads reduce locally.
__global__ void kLayerNormStats(const float* src, float* mean, float* variance, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute Sum (Mean)
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_sum += src[row * cols + col];
    }

    // Block Reduction for Sum
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_mean = sdata[0] / cols;
    if (threadIdx.x == 0) mean[row] = row_mean;
    __syncthreads(); // Wait for mean to be written

    // Compute Variance
    // Recalculate row_mean from shared memory to ensure all threads have it
    row_mean = sdata[0] / cols; 
    
    float local_diff_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float diff = src[row * cols + col] - row_mean;
        local_diff_sq += diff * diff;
    }

    // Block Reduction for Variance
    sdata[threadIdx.x] = local_diff_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) variance[row] = sdata[0] / cols;
}

// 2. Forward Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void kLayerNormForward(const float* src, float* dest, const float* mean, const float* var, 
                                  const float* gamma, const float* beta, 
                                  int rows, int cols, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    int col = idx % cols;

    float mu = mean[row];
    float sigma = sqrtf(fmaxf(0.0f, var[row]) + epsilon);
    float x_hat = (src[idx] - mu) / sigma;

    dest[idx] = x_hat * gamma[col] + beta[col];
}

// 3. Backward Pass (Fused Gradient Calculation)
// Computes dGamma, dBeta, and dInput in one go is hard, 
// so we split: Step A (Accumulate Params), Step B (Compute dInput)

__global__ void kLayerNormBackwardParams(const float* d_out, const float* src, const float* mean, const float* var,
                                         float* d_gamma, float* d_beta, 
                                         int rows, int cols, float epsilon) {
    // We parallelize over COLUMNS (Features). 
    // Each thread sums down the rows for its specific feature column.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float d_g_sum = 0.0f;
    float d_b_sum = 0.0f;

    for (int row = 0; row < rows; ++row) {
        int idx = row * cols + col;
        float mu = mean[row];
        float sigma = sqrtf(var[row] + epsilon);
        float x_hat = (src[idx] - mu) / sigma;

        d_g_sum += d_out[idx] * x_hat;
        d_b_sum += d_out[idx];
    }
    
    d_gamma[col] = d_g_sum;
    d_beta[col] = d_b_sum;
}

__global__ void kLayerNormBackwardInput(const float* d_out, const float* src, const float* mean, const float* var,
                                        const float* gamma, float* d_in, 
                                        int rows, int cols, float epsilon) {
    // One block per ROW.
    int row = blockIdx.x;
    if (row >= rows) return;

    // Calculate reduction terms for this row
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    float mu = mean[row];
    float inv_sigma = 1.0f / sqrtf(var[row] + epsilon);

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        float x_hat = (src[idx] - mu) * inv_sigma;
        float dy = d_out[idx]; // Note: Logic usually requires dy * gamma here? 
                               // Actually standard derivation: 
                               // dl/dxhat = dl/dy * gamma. 
        
        sum_dy += dy * gamma[col]; 
        sum_dy_xhat += (dy * gamma[col]) * x_hat;
    }

    // Block Reduce
    __shared__ float sdata_dy[BLOCK_SIZE];
    __shared__ float sdata_dy_xhat[BLOCK_SIZE];
    sdata_dy[threadIdx.x] = sum_dy;
    sdata_dy_xhat[threadIdx.x] = sum_dy_xhat;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_dy[threadIdx.x] += sdata_dy[threadIdx.x + s];
            sdata_dy_xhat[threadIdx.x] += sdata_dy_xhat[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float row_sum_dy = sdata_dy[0];
    float row_sum_dy_xhat = sdata_dy_xhat[0];

    // Compute Gradient per element
    float term1 = 1.0f / cols;
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        float x_hat = (src[idx] - mu) * inv_sigma;
        float dy = d_out[idx] * gamma[col]; // This is dl/dxhat contribution

        // Standard LN Gradient Formula:
        // dx = (1/N) * inv_sigma * (N * dy - sum_dy - x_hat * sum_dy_xhat)
        // Here dy is actually (d_out * gamma)
        
        float val = term1 * inv_sigma * ( (cols * dy) - row_sum_dy - (x_hat * row_sum_dy_xhat) );
        d_in[idx] = val;
    }
}

// Compute Max per row (for numerical stability)
__global__ void kRowMax(const float* src, float* max_vals, int rows, int cols) {
    int row = blockIdx.x; // One block per row
    if (row >= rows) return;

    float local_max = -1e9f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = src[row * cols + col];
        if (val > local_max) local_max = val;
    }

    // Block-level reduction using shared memory
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) max_vals[row] = sdata[0];
}

// Compute Sum of Exponentials per row
__global__ void kRowSumExp(const float* src, const float* max_vals, float* sum_vals, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float row_max = max_vals[row];
    float local_sum = 0.0f;

    // Numerical Guard: If the row_max is a masking value (e.g., -1e20f), 
    // the entire row is likely invalid/masked.
    if (row_max < -1e10f) {
        if (threadIdx.x == 0) sum_vals[row] = 1.0f; // Prevent div by zero later
        return;
    }

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        // Only compute exp if the value is within a reasonable range of the max
        float val = src[row * cols + col] - row_max;
        if (val > -80.0f) { // Standard range for float expf stability
            local_sum += expf(val);
        }
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Ensure sum_vals never becomes exactly 0
        sum_vals[row] = fmaxf(sdata[0], 1e-30f);
    }
}


// Final Softmax: Exp(x - max) / Sum
// Fixed: Handles cases where max_val is -Infinity or sum is 0 to prevent NaNs
__global__ void kApplySoftmax(const float* src, float* dest, const float* max_vals, const float* sum_vals, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        float max_val = max_vals[row];
        float sum = sum_vals[row];

        // Safety Check: If the row is masked (max is -Infinity) or Sum is 0
        // Match the CPU logic: return uniform distribution (1.0 / cols)
        if (max_val < -1e8f || sum <= 1e-20f) {
            dest[idx] = 1.0f / cols;
        } 
        else {
            // Standard Softmax Calculation
            float val = expf(src[idx] - max_val);
            dest[idx] = val / sum; 
        }
    }
}

// Softmax Backward: d_in = softmax * (d_out - sum(d_out * softmax))
// This kernel calculates the dot product (sum(d_out * softmax)) per row
__global__ void kSoftmaxGradDot(const float* grad_output, const float* softmax_output, float* dot_products, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float local_dot = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        local_dot += grad_output[idx] * softmax_output[idx];
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) dot_products[row] = sdata[0];
}

// Softmax Backward Final Calculation
__global__ void kApplySoftmaxBackward(const float* grad_output, const float* softmax_output, const float* dot_products, float* grad_input, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        float s = softmax_output[idx];
        float dy = grad_output[idx];
        float dot = dot_products[row];
        grad_input[idx] = s * (dy - dot);
    }
}

// Sums columns: Collapses an (rows, cols) matrix into a (1, cols) vector
// Used for Bias Gradients in vectorized layers
__global__ void kSumColumns(const float* src, float* dest, int rows, int cols) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < rows * cols) {
        int col = idx % cols;
        // Atomic add is simplest implementation for variable row sizes
        atomicAdd(&dest[col], src[idx]);
    }
}

// Inside Kernel 1 (Element-wise Addition) - Update to this:
__global__ void kAddBroadcast(const float* a, const float* b, float* c, int rows, int cols, int b_rows) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < rows * cols) {
        int col = idx % cols; // Column index
        // If b is a row vector (1, cols), use b[col]. If b is full matrix, use b[idx]
        float b_val = (b_rows == 1) ? b[col] : b[idx];
        c[idx] = a[idx] + b_val;
    }
}

// -------------------------------------------------------------------------
// Kernel: Parallel Sum of Squares (Reduction)
// -------------------------------------------------------------------------
__global__ void kSumSquares(const float* __restrict__ gradients, float* total_sum_sq, int n) {
    // Shared memory for block-level reduction
    __shared__ float cache[BLOCK_SIZE];

    int tid = threadIdx.x;
    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp_sum = 0.0f;

    // Grid-stride loop: Allows kernel to handle vectors larger than the grid
    while (grid_idx < n) {
        float val = gradients[grid_idx];
        temp_sum += val * val;
        grid_idx += gridDim.x * blockDim.x;
    }

    cache[tid] = temp_sum;
    __syncthreads();

    // Block reduction: parallel sweep to sum cache entries
    // This reduces O(N) complexity to O(log N) within the block
    int i = BLOCK_SIZE / 2;
    while (i != 0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    // First thread of each block adds its partial sum to the global accumulator
    if (tid == 0) {
        atomicAdd(total_sum_sq, cache[0]);
    }
}

// Splits a large gradient matrix into smaller head-specific matrices
__global__ void kSplitGradients(const float* src, float** head_grad_ptrs, int num_heads, int head_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total_tokens * num_heads * head_dim;
    
    if (idx < total_elements) {
        int token_idx = idx / (num_heads * head_dim);
        int feature_idx = idx % (num_heads * head_dim);
        int head_idx = feature_idx / head_dim;
        int head_feature_idx = feature_idx % head_dim;
        
        head_grad_ptrs[head_idx][token_idx * head_dim + head_feature_idx] = src[idx];
    }
}

// -------------------------------------------------------------------------
// Kernel: Apply Scaling
// -------------------------------------------------------------------------
__global__ void kApplyScale(float* gradients, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        gradients[i] *= scale;
    }
}

__global__ void kSetDiagonal(float* out, int rows, int cols, float val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int limit = (rows < cols) ? rows : cols; // min(rows, cols)
    
    if (idx < limit) {
        // Set M[idx][idx] = val
        out[idx * cols + idx] = val;
    }
}

__global__ void kReset(float* out, int rows, int cols, float val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int limit = rows * cols;
    
    if (idx < limit) {
        out[idx] = val;
    }
}

__global__ void kDropout(float* out, int size, float p, unsigned int seed, int step) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        // Simple hash for randomness on GPU
        unsigned int hash = idx ^ seed ^ step;
        hash = hash * 1664525u + 1013904223u;
        float random = (float)(hash & 0xFFFFFF) / 16777216.0f;
        
        out[idx] = (random < p) ? 0.0f : 1.0f;
    }
}

__global__ void kGelu(const float* in, float* out, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        float x = in[idx];
        // Standard Tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
        float x_cubed = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void kGeluBackward(const float* inputs, const float* gradients, float* out, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        float x = inputs[idx];
        float g = gradients[idx];
        
        // Precise derivative logic to match your CPU code
        // cdf = 0.5 * (1 + erf(x / sqrt(2)))
        // pdf = (1 / sqrt(2pi)) * exp(-0.5 * x^2)
        const float SQRT_2 = 1.41421356237f;
        const float INV_SQRT_2PI = 0.3989422804f;
        
        float cdf = 0.5f * (1.0f + erf(x / SQRT_2));
        float pdf = INV_SQRT_2PI * expf(-0.5f * x * x);
        float term2 = x * pdf;
        if (isinf(x)) term2 = 0.0f; 
        float derivative = cdf + term2;
        out[idx] = g * derivative;
    }
}


// Element-wise Addition
__global__ void kAdd(const float* a, const float* b, float* c, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Matrix Transpose
__global__ void kTranspose(const float* in, float* out, int rows, int cols) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (x < cols && y < rows) {
        // out[x][y] = in[y][x]
        out[x * rows + y] = in[y * cols + x];
    }
}

// Banded Matrix Multiplication
__global__ void kBandedMul(const float* A, const float* B, float* C, 
                           int rows, int cols, int common, int window) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row < rows && col < cols) {
        // Initialize with -Infinity if masked? 
        // No, standard mul starts at 0. The masking happens later or implicit via loop.
        float sum = 0.0f;
        int start_j = max(0, row - window);
        int end_j = min(cols, row + window + 1);

        // If this thread (row, col) is outside the band, leave it (or set to -inf if requested)
        if (col >= start_j && col < end_j) {
            for (int k = 0; k < common; ++k) {
                sum += A[row * common + k] * B[k * cols + col];
            }
            C[row * cols + col] = sum;
        } else {
            // If outside band, set to -infinity
             C[row * cols + col] = -INFINITY;
        }
    }
}

// Corrected kScaleMask for Batched Stride (Tall & Narrow Matrix)
__global__ void kScaleMask(const float* in, float* out, int rows, int cols, float scale, int valid_seq_len, int seq_len) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        
        // Calculate LOCAL row index (0 to 63) within the sequence
        int local_row = row % seq_len;
        
        // 1. Causal Masking: Mask if column is ahead of the current token
        // 2. Padding Masking: Mask if column is past valid length
        if (col > local_row || col >= valid_seq_len) {
            out[idx] = -1e20f; 
        } else {
            out[idx] = in[idx] * scale;
        }
    }
}

__global__ void kScale(const float* in, float* out, int size, float scale) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        out[idx] = in[idx] * scale;
    }
}

// Element-wise Product
__global__ void kElementWiseMul(const float* a, const float* b, float* c, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// --- KERNEL WRAPPERS (Must be outside the template class) ---
void launch_set_diagonal_kernel(float* data, int rows, int cols, float val) {
    int min_dim = (rows < cols) ? rows : cols;
    int threads = 256;
    int blocks = (min_dim + threads - 1) / threads;
    kSetDiagonal<<<blocks, threads>>>(data, rows, cols, val);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::stringstream ss;
      ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
}

void launch_reset_kernel(float* data, int rows, int cols, float val) {
    if (data == nullptr) {
      throw std::runtime_error("FATAL: launch_reset_kernel received a null pointer! "
                                 "Check hipMalloc calls upstream.");
    }
    int total = rows * cols; 
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    // Bypass the <<<>>> syntax using the explicit HIP launch macro
    hipLaunchKernelGGL(kReset, dim3(blocks), dim3(threads), 0, 0, data, rows, cols, val);
    
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::stringstream ss;
      ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
}


