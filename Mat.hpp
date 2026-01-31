#pragma once
#include "Common.hpp"
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

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        // Only compute exp if the value is within a reasonable range of the max
        float val = src[row * cols + col] - row_max;
        if (val > -50.0f) { // Standard range for float expf stability
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
        sum_vals[row] = fmaxf(sdata[0], 1e-25f);
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


// Static handle for rocBLAS to avoid recreating it constantly
static rocblas_handle rb_handle = nullptr;
static std::mutex rb_init_mtx;

static void init_rb_handle() {
  std::lock_guard<std::mutex> lock(rb_init_mtx);
  if (!rb_handle) {
    rocblas_create_handle(&rb_handle);
  }
}

static void destroy_rb_handle() {
    if (rb_handle) {
        rocblas_destroy_handle(rb_handle);
        rb_handle = nullptr;
    }
}

class GPUMemoryArena {
    void* base_ptr = nullptr;
    size_t total_size = 0;
    // Use atomic for lock-free thread safety
    std::atomic<size_t> offset{0}; 

public:
    void init(size_t size_bytes = 1024 * 1024 * 512) {
        if (base_ptr) return; 
        total_size = size_bytes;
        hipError_t err = hipMalloc(&base_ptr, total_size);
        if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
        }
        offset.store(0);
        err = hipMemset(base_ptr, 0, total_size);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipMemset failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
    }

    void* allocate(size_t size) {
        // 1. Align to 256 bytes for coalesced access (keeps performance high)
        size_t aligned_size = (size + 255) / 256 * 256;

        // 2. Lock-Free Reservation: atomic fetch_add is much faster than mutex
        size_t current_offset = offset.fetch_add(aligned_size, std::memory_order_relaxed);

        if (current_offset + aligned_size > total_size) {
            std::stringstream ss;
            ss << "GPU Arena Out of Memory! Used: " << current_offset 
               << " Requested: " << size << " (Aligned: " << aligned_size << ")";
            throw std::runtime_error(ss.str());
        }

        return (char*)base_ptr + current_offset;
    }

    void reset() {
        // Resetting is just a simple atomic store
        offset.store(0, std::memory_order_relaxed);
    }

    void free_all() {
        if (base_ptr) {
            hipError_t err = hipFree(base_ptr);
            if (err != hipSuccess) {
              std::stringstream ss;
              ss << "hipFree failed: " << hipGetErrorString(err);
              throw std::runtime_error(ss.str());
            }
 
            base_ptr = nullptr;
        }
    }
    size_t get_offset() const {
        return offset.load(std::memory_order_relaxed);
    }

    void set_offset(size_t new_offset) {
        offset.store(new_offset, std::memory_order_relaxed);
    }
};

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
    int total = rows * cols; // Reset the whole matrix, not just diagonal
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kReset<<<blocks, threads>>>(data, rows, cols, val);
    hipError_t err = hipDeviceSynchronize();

    if (err != hipSuccess) {
      std::stringstream ss;
      ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
}

// Global instance
static GPUMemoryArena global_arena;
template <typename T>
class Mat {

    class MatRow {
        T * row_start;
        public:
        MatRow(T * row_start) : row_start(row_start) {}
        T & operator[](size_t col) { return row_start[col]; }
        const T & operator[](size_t col) const { return row_start[col]; }
    };

    int rows;
    int cols;
public:
    T* d_data = nullptr; 
    std::vector<T> data;
private:
    bool cpu_dirty = true;
    bool from_pool = false; 

public:
    void dirty() { cpu_dirty = true;}
    static bool enable_arena; 
    // --- CONSTRUCTORS ---
    Mat() : rows(0), cols(0) { init_rb_handle(); reset(); }
    Mat(int rows, int cols) : rows(rows), cols(cols) {
        from_pool = false; 
        init_rb_handle();
        allocate_device_memory();
        reset();
        cpu_dirty = false;
    }

    Mat(int rows, int cols, const std::vector<T> & input_data) : rows(rows), cols(cols) {
        from_pool = false; 
        init_rb_handle();
        allocate_device_memory();
        reset();
        data = input_data;
        hipError_t err = hipMemcpy(d_data, data.data(), rows * cols * sizeof(T), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
          std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("hipMemcpy failed");
        }
        cpu_dirty = false;
    }

    // Copy Constructor (Deep Copy)
    Mat(const Mat& other) : rows(other.rows), cols(other.cols) {
        from_pool = false; 
        init_rb_handle();
        allocate_device_memory();
        reset();
        if (!other.cpu_dirty || other.d_data == nullptr) {
          data = other.data; 
        }
        cpu_dirty = other.cpu_dirty;
        if (other.d_data) {
          hipError_t err = hipMemcpy(d_data, other.d_data, rows * cols * sizeof(T), hipMemcpyDeviceToDevice);
          if (err != hipSuccess) {
            std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
            throw std::runtime_error("hipMemcpy failed");
          }
        }
    }

    // --- MOVE SEMANTICS (CRITICAL FOR MEMORY FIX) ---
    // Steals resources from 'other' instead of copying.
    Mat(Mat&& other) 
        : rows(other.rows), cols(other.cols), d_data(other.d_data), 
          data(std::move(other.data)), cpu_dirty(other.cpu_dirty), from_pool(other.from_pool) {
        other.d_data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.from_pool = false; // Prevent double free
    }

    void copy_from(const Mat<T>& other) {
      if (this == &other) return;
      hipError_t err = hipMemcpy(d_data, other.d_data, rows * cols * sizeof(T), hipMemcpyDeviceToDevice);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }

      data = other.data;
      cpu_dirty = true; // Mark CPU data as out of sync
    }

    // ----------------------------------------------------------------------
    // BATCHED MATRIX OPERATIONS (Fixed for Row-Major / RocBLAS compatibility)
    // ----------------------------------------------------------------------

    // C = A * B (Batched)
    // Logic: C_row = A_row * B_row  <=>  C_col = B_col * A_col
    // RocBLAS (Col-Major): Pass B as Arg1, A as Arg2. No Transpose.
    void batched_mul(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
        int M = rows / batch_count;     // Rows of A
        int K = cols;                   // Cols of A / Rows of B
        int N = rhs.cols;               // Cols of B
/*
        if (result.rows != rows || result.cols != N) {
             result.assign(rows, N); // Auto-resize if needed
        }
*/
        float alpha = 1.0f;
        float beta = 0.0f;

        // rocBLAS call: C_col (NxM) = B_col (NxK) * A_col (KxM)
        rocblas_sgemm_strided_batched(rb_handle,
            rocblas_operation_none, rocblas_operation_none,
            N, M, K,
            &alpha,
            rhs.d_data, N, stride_b,      // Arg1: B
            this->d_data, K, stride_a,    // Arg2: A
            &beta,
            result.d_data, N, stride_c,   // Result: C
            batch_count
        );
        result.cpu_dirty = true;
    }

    // C = A * B^T (Batched)
    // Used for: Q * K^T
    // Logic: C_col = (A * B^T)^T = B * A^T
    // RocBLAS: Arg1=B, Arg2=A. Op1=Trans, Op2=NoTrans.
    void batched_mul_transpose(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
        int M = rows / batch_count;     // Rows of A (Seq)
        int K = cols;                   // Cols of A (Head)
        int N = rhs.rows / batch_count; // Rows of B (Seq) -> Cols of B^T
        // Result should be (Batch*Seq, Seq)
/*
        if (result.rows != rows || result.cols != N) {
            result.assign(rows, N);
        }
*/
        float alpha = 1.0f;
        float beta = 0.0f;

        // rocBLAS call: C_col (NxM) = B_col^T (NxK) * A_col (KxM) ?? 
        // Wait: B_col is (KxN). B_col^T is (NxK).
        rocblas_sgemm_strided_batched(rb_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            N, M, K,
            &alpha,
            rhs.d_data, K, stride_b,      // Arg1: B (Transposed)
            this->d_data, K, stride_a,    // Arg2: A (No Transpose)
            &beta,
            result.d_data, N, stride_c,
            batch_count
        );
        result.cpu_dirty = true;
    }

    // C = A^T * B (Batched)
    // Used for: dV = S^T * dO, dK = dS^T * Q
    // Logic: C_col = (A^T * B)^T = B^T * A
    // RocBLAS: Arg1=B, Arg2=A. Op1=None, Op2=Trans.
    void batched_mul_lhs_transpose(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
        int K = rows / batch_count;    // Rows of A (Seq). A^T has K cols.
        int M = cols;                  // Cols of A (Head/Seq). A^T has M rows.
        int N = rhs.cols;              // Cols of B.

        // Result should be (Batch*M, N)
        // Note: 'rows' of 'this' is Batch*Seq. But 'this' is being transposed.
        // So result rows = Batch * M.
/*        if (result.rows != batch_count * M || result.cols != N) {
             result.assign(batch_count * M, N);
        }
*/
        float alpha = 1.0f;
        float beta = 0.0f;

        // rocBLAS call: C_col (NxM) = B_col (NxK) * A_col^T (KxM)
        // Note: A_col is (MxK). A_col^T is (KxM).
        rocblas_sgemm_strided_batched(rb_handle,
            rocblas_operation_none, rocblas_operation_transpose,
            N, M, K,
            &alpha,
            rhs.d_data, N, stride_b,      // Arg1: B
            this->d_data, M, stride_a,    // Arg2: A (Transposed)
            &beta,
            result.d_data, N, stride_c,
            batch_count
        );
        result.cpu_dirty = true;
    }

    static void computeLogitGradients(const Mat<float>& predictions, const Mat<float>& targets, Mat<float>& grad_output) {
      int rows = predictions.get_rows();
      int cols = predictions.get_cols();
      int size = rows * cols;

      if (targets.get_rows() != rows || targets.get_cols() != cols) {
        throw std::runtime_error("Dimension mismatch in computeLogitGradients targets");
      }

      // Ensure output matrix is ready (uses Arena if enabled)
      if (grad_output.get_rows() != rows || grad_output.get_cols() != cols) {
        grad_output.assign(rows, cols);
      }

      int threads = 256;
      int blocks = (size + threads - 1) / threads;

      kComputeLogitGradients<<<blocks, threads>>>(
          predictions.d_data, 
          targets.d_data, 
          grad_output.d_data, 
          size
          );

      grad_output.dirty(); // Mark that CPU data is now out of sync
    }

    static float softCrossEntropyLoss(const Mat<float>& predicted_probs, const Mat<float>& soft_labels, int batch_size, int seq_length) {
      int total_elements = predicted_probs.get_rows() * predicted_probs.get_cols();
      float h_loss = 0.0f;
      float* d_loss_ptr = nullptr;

      // 1. Memory Allocation via Arena or hipMalloc
      if (Mat<float>::enable_arena) {
        d_loss_ptr = (float*)global_arena.allocate(sizeof(float));
      } else {
        hipError_t err = hipMalloc(&d_loss_ptr, sizeof(float));
        if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
        }
      }

      // 2. Initialize the GPU accumulator to zero
      hipError_t err = hipMemset(d_loss_ptr, 0, sizeof(float));
      if (err != hipSuccess) {
        std::stringstream ss;
        ss << "hipMemset failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
      }

      // 3. Launch Kernel
      int threads = 256;
      int blocks = (total_elements + threads - 1) / threads;
      float epsilon = 1e-15f;

      kSoftCrossEntropy<<<blocks, threads>>>(
          predicted_probs.d_data, 
          soft_labels.d_data, 
          d_loss_ptr, 
          total_elements, 
          epsilon
          );

      // 4. Synchronize and copy back the single float result
      err = hipMemcpy(&h_loss, d_loss_ptr, sizeof(float), hipMemcpyDeviceToHost);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }
      // 5. Cleanup if not using arena
      if (!Mat<float>::enable_arena) {
        hipError_t err = hipFree(d_loss_ptr);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }

      // Return the averaged loss
      return h_loss / (float)(batch_size * seq_length);
    }

    static void split_gradients(Mat<T>& src, const std::vector<Mat<T>>& head_grads) {
      int num_heads = head_grads.size();
      int head_dim = head_grads[0].cols;
      int total_tokens = head_grads[0].rows;
      int total_elements = total_tokens * num_heads * head_dim;

      T** d_head_ptrs = nullptr;
      if (enable_arena) {
        d_head_ptrs = (T**)global_arena.allocate(num_heads * sizeof(T*));
      } else {
        hipError_t err = hipMalloc(&d_head_ptrs, num_heads * sizeof(T*));
        if (err != hipSuccess) {
          std::cerr << "HipMalloc failed (" << sizeof(T*) * num_heads << " bytes): " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("GPU OOM or Error");
        }
      }

      std::vector<T*> h_ptrs(num_heads);
      for(int i = 0; i < num_heads; ++i) h_ptrs[i] = head_grads[i].d_data;

      hipError_t err = hipMemcpy(d_head_ptrs, h_ptrs.data(), num_heads * sizeof(T*), hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }

      int threads = 256;
      int blocks = (total_elements + threads - 1) / threads;

      kSplitGradients<<<blocks, threads>>>(src.d_data, d_head_ptrs, num_heads, head_dim, total_tokens);

      if (!enable_arena && d_head_ptrs != nullptr) {
        hipError_t err = hipDeviceSynchronize();

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
        err = hipFree(d_head_ptrs);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }
    }

    static void concatenate_heads(const std::vector<Mat<T>>& heads, Mat<T>& dest) {
      int num_heads = heads.size();
      int head_dim = heads[0].cols;
      int total_tokens = heads[0].rows;
      int total_elements = total_tokens * num_heads * head_dim;

      // Allocate a small temporary array of pointers ON THE GPU via the Arena
      T** d_head_ptrs = nullptr;
      if (enable_arena) {
        d_head_ptrs = (T**)global_arena.allocate(num_heads * sizeof(T*));
      } else {
        hipError_t err = hipMalloc(&d_head_ptrs, num_heads * sizeof(T*));
        if (err != hipSuccess) {
          std::cerr << "HipMalloc failed (" << sizeof(T*) * num_heads << " bytes): " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("GPU OOM or Error");
        }
      }

      // Prepare the pointer list on CPU
      std::vector<T*> h_ptrs(num_heads);
      for(int i = 0; i < num_heads; ++i) h_ptrs[i] = heads[i].d_data;

      // Copy pointer list to GPU
      hipError_t err = hipMemcpy(d_head_ptrs, h_ptrs.data(), num_heads * sizeof(T*), hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }

      int threads = 256;
      int blocks = (total_elements + threads - 1) / threads;

      kConcatenateHeads<<<blocks, threads>>>(d_head_ptrs, dest.d_data, num_heads, head_dim, total_tokens);
      dest.dirty();

      if (!enable_arena && d_head_ptrs != nullptr) {
        hipError_t err = hipDeviceSynchronize();

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
        err = hipFree(d_head_ptrs);

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }
    }

    // Standard Assignment (Copy)
    Mat & operator=(const Mat & other) {
      if (this == &other) return *this;

      // We trust the logic that 'this' pointer is still valid within the current Arena step.
      bool can_reuse = (d_data != nullptr) && (rows == other.rows) && (cols == other.cols);

      if (can_reuse) {
        hipError_t err = hipMemcpy(d_data, other.d_data, rows * cols * sizeof(T), hipMemcpyDeviceToDevice);
        if (err != hipSuccess) {
          std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("hipMemcpy failed");
        }


        if (!other.cpu_dirty || other.d_data == nullptr) {
          data = other.data;
        } else {
          std::vector<T>().swap(data);
        }
        cpu_dirty = other.cpu_dirty;
        return *this;
      }
     
      if (d_data && !from_pool) {
        hipError_t err = hipFree(d_data);

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }
      rows = other.rows;
      cols = other.cols;
      // Only copy CPU vector if necessary
      if (!other.cpu_dirty || other.d_data == nullptr) {
        data = other.data;
      } else {
        std::vector<T>().swap(data); // Ensure we start fresh
      }

      cpu_dirty = other.cpu_dirty;
      allocate_device_memory();
      if (other.d_data) {
        hipError_t err = hipMemcpy(d_data, other.d_data, rows * cols * sizeof(T), hipMemcpyDeviceToDevice);
        if (err != hipSuccess) {
          std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("hipMemcpy failed");
        }
      }
      return *this;
    }

    ~Mat() {
      if (d_data && !from_pool) {
        hipError_t err =  hipFree(d_data);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
        }
      }
        release_cpu();
    }

    void allocate_device_memory() {
        size_t size = rows * cols * sizeof(T);
        if (size == 0) { d_data = nullptr; return; }
        if (enable_arena) {
            d_data = (T*)global_arena.allocate(size);
            from_pool = true;
        } else {
            hipError_t err = hipMalloc(&d_data, size);
            if (err != hipSuccess) {
                std::cerr << "HipMalloc failed (" << size << " bytes): " << hipGetErrorString(err) << std::endl;
                throw std::runtime_error("GPU OOM or Error");
            }
            from_pool = false;
        }
    }

    // --- MEMORY OPTIMIZATION ---
    // Call this to free CPU memory if you only need the data on GPU
    void release_cpu() {
        std::vector<T>().swap(data); 
        cpu_dirty = true; // Mark that CPU is empty/invalid
    }

    void to_cpu() {
      if (cpu_dirty && d_data) {
        hipError_t err = hipDeviceSynchronize();

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }

        if (data.size() != rows * cols) data.resize(rows * cols);
        err = hipMemcpy(data.data(), d_data, rows * cols * sizeof(T), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
          std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("hipMemcpy failed");
        }
        cpu_dirty = false;
      }
    }

    void to_gpu() {
      if (d_data && data.size() == rows * cols) {
        hipError_t err = hipMemcpy(d_data, data.data(), rows * cols * sizeof(T), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
          std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
          throw std::runtime_error("hipMemcpy failed");
        }

      }
    }
    
    // Raw Accessors
    MatRow operator[](size_t row) { to_cpu(); return MatRow(data.data() + row * cols); }

    std::vector<T> & raw() { 
 //     if(data.size() != rows*cols) data.resize(rows*cols); 
      to_cpu(); 
      return data; 
    }
    
    void set_raw(const std::vector<T> & tmp) {
        data = tmp; // Copy input
        to_gpu();   // Upload
    }
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }

    void mul(const Mat<T> & rhs, Mat<T> & result) const {
        assert(cols == rhs.rows);
        // The result of (rows x cols) * (cols x rhs.cols) is (rows x rhs.cols)
        if (result.rows != rows || result.cols != rhs.cols) {
          throw std::runtime_error("MUL - invalid size");
        }
        float alpha = 1.0f; float beta = 0.0f;
        rocblas_sgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                      rhs.cols, rows, cols, &alpha, rhs.d_data, rhs.cols,
                      d_data, cols, &beta, result.d_data, rhs.cols);
        result.cpu_dirty = true;
    }
    
    void add(const Mat<T> & rhs, Mat<T> & result) const {
      // CASE 1: Exact Match
      if (rows == rhs.rows && cols == rhs.cols) {
          int size = rows * cols;
          int threads = 256; int blocks = (size + threads - 1) / threads;
          kAdd<<<blocks, threads>>>(d_data, rhs.d_data, result.d_data, size);
      }
      // CASE 2: Broadcast RHS (1, Cols) onto LHS (Rows, Cols)
      // This allows: Matrix(N, E) + Bias(1, E)
      else if (rhs.rows == 1 && cols == rhs.cols) {
          int size = rows * cols;
          int threads = 256; int blocks = (size + threads - 1) / threads;
          kAddBroadcast<<<blocks, threads>>>(d_data, rhs.d_data, result.d_data, rows, cols, rhs.rows);
      }
      else {
          throw std::runtime_error("ADD - invalid dimension mismatch");
      }
      result.cpu_dirty = true;
    }

     // Collapses (rows, cols) -> (1, cols) by summing down columns
    void sum_rows(Mat<T> & result) {
        if (result.rows != 1 || result.cols != cols) {
            throw std::runtime_error("sum_rows - result must be (1, cols)");
        }
        
        // 1. Zero out the destination first
        hipError_t err = hipMemset(result.d_data, 0, cols * sizeof(T));

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipMemset failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
        
        // 2. Launch Kernel
        int size = rows * cols;
        int threads = 256; 
        int blocks = (size + threads - 1) / threads;
        kSumColumns<<<blocks, threads>>>(d_data, result.d_data, rows, cols);
        
        result.cpu_dirty = true;
    }

    void transpose(Mat<T> & out) const {
      // Ensure output dimensions match
      if (out.rows != cols || out.cols != rows) {
        //out.assign(cols, rows); // Reallocate only if size changes
        throw std::runtime_error("TRANSPOSE - invalid size");
      }
      // Launch kernel writing directly to 'out.d_data'
      // No new hipMalloc calls will happen if dimensions are stable!
      dim3 threads(16, 16); 
      dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
      kTranspose<<<blocks, threads>>>(d_data, out.d_data, rows, cols);
      out.cpu_dirty = true;
    }
    
    void element_wise_product(const Mat<T>& rhs, Mat<T>& result) {
        int size = rows * cols;
        if (rows != rhs.rows || cols != rhs.cols) {
          throw std::runtime_error("EWP - invalid size");
        }
        if (rows != result.rows || cols != result.cols) {
          throw std::runtime_error("EWP result - invalid size");
        }

        int threads = 256; int blocks = (size + threads - 1) / threads;
        kElementWiseMul<<<blocks, threads>>>(d_data, rhs.d_data, result.d_data, size);
        result.cpu_dirty = true;
    }
    
    void scale(T scale_val, Mat<T> & result) {
        int size = rows * cols;
        if (rows != result.rows || cols != result.cols) {
          throw std::runtime_error("scale result - invalid size");
        }

        int threads = 256; int blocks = (size + threads - 1) / threads;
        kScale<<<blocks, threads>>>(d_data, result.d_data, size, scale_val);
        result.cpu_dirty = true;
    }

    void scale_mask(T scale, int valid_seq_len, int seq_len, Mat<T> & result) {
        if (rows != result.rows || cols != result.cols) {
          throw std::runtime_error("scale result - invalid size");
        }
        dim3 threads(16, 16); dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
        kScaleMask<<<blocks, threads>>>(d_data, result.d_data, rows, cols, scale, valid_seq_len, seq_len);
        result.cpu_dirty = true;
    }

    // GELU OPERATIONS
    void gelu(Mat<T> & result) {
        int size = rows * cols;
        if (rows != result.rows || cols != result.cols) {
          throw std::runtime_error("scale result - invalid size");
        }
        int threads = 256; int blocks = (size + threads - 1) / threads;
        kGelu<<<blocks, threads>>>(d_data, result.d_data, size);
        result.cpu_dirty = true;
    }

    // result = gradient * gelu_derivative(input)
    void gelu_backward(const Mat<T>& gradients, Mat<T> & result) {
        if (rows != gradients.rows || cols != gradients.cols) {
          throw std::runtime_error("gelu_back - invalid size");
        }
        if (result.rows != rows || result.cols != cols) {
          throw std::runtime_error("gelu_back result - invalid size");
        }

        int size = rows * cols;
        int threads = 256; int blocks = (size + threads - 1) / threads;
        // this->d_data is the Input (Z), gradients.d_data is Upstream Grad
        kGeluBackward<<<blocks, threads>>>(d_data, gradients.d_data, result.d_data, size);
        result.cpu_dirty = true;
    }

    void softmax(Mat<T>& result) {
        if (result.rows != rows || result.cols != cols) {
            result.assign(rows, cols);
        }

        float* d_max;
        float* d_sum;

        // Use the Arena for temporary row-wise statistics
        if (enable_arena) {
            d_max = (float*)global_arena.allocate(rows * sizeof(float));
            d_sum = (float*)global_arena.allocate(rows * sizeof(float));
        } else {
          // Fallback if arena is disabled
          hipError_t err = hipMalloc(&d_max, rows * sizeof(float));

          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }

          err = hipMalloc(&d_sum, rows * sizeof(float));
          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }

        }

        // Launch kernels using the arena-allocated pointers
        kRowMax<<<rows, BLOCK_SIZE>>>(d_data, d_max, rows, cols);
        kRowSumExp<<<rows, BLOCK_SIZE>>>(d_data, d_max, d_sum, rows, cols);

        int total_elements = rows * cols;
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kApplySoftmax<<<blocks, BLOCK_SIZE>>>(d_data, result.d_data, d_max, d_sum, rows, cols);

        // Only free if we didn't use the arena pool
        if (!enable_arena) {
          hipError_t err = hipFree(d_max);
          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "hipFree failed: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }
          err = hipFree(d_sum);

          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "hipFree failed: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }
        }

        result.dirty();
    }

    // New Helper for GPU Backward Pass
    void softmax_backward(const Mat<T>& softmax_output, Mat<T>& grad_input) {
        // This 'Mat' (this) is the upstream gradient (d_out)
        // softmax_output is the output of the forward softmax
        
        if (rows != softmax_output.rows || cols != softmax_output.cols) throw std::runtime_error("Size mismatch in softmax_backward");
        if (grad_input.rows != rows || grad_input.cols != cols) grad_input.assign(rows, cols);

        float* d_dot;
        if (enable_arena) {
          d_dot = (float*)global_arena.allocate(rows * sizeof(float));
        } else {
          hipError_t err = hipMalloc(&d_dot, rows * sizeof(float));

          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }

        }

        // 1. Calculate Dot Product per row
        kSoftmaxGradDot<<<rows, BLOCK_SIZE>>>(this->d_data, softmax_output.d_data, d_dot, rows, cols);

        // 2. Apply final gradient formula
        int total_elements = rows * cols;
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kApplySoftmaxBackward<<<blocks, BLOCK_SIZE>>>(this->d_data, softmax_output.d_data, d_dot, grad_input.d_data, rows, cols);

        if (!enable_arena) {
          hipError_t err = hipFree(d_dot);

          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "hipFree failed: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }
        }
        grad_input.cpu_dirty = true;
    }

    static void create_dropout_mask(int rows, int cols, float p, Mat<T> & M) {
      if (M.rows != rows || M.cols != cols) {
        throw std::runtime_error("drop_out_mask result - invalid size");
      }
      static std::atomic<int> step_counter(0);
      step_counter++;
      int size = rows * cols;
      int threads = 256;
      int blocks = (size + threads - 1) / threads;
      kDropout<<<blocks, threads>>>(M.d_data, size, p, seed, (int)step_counter);
    }
   
    // Static Factory: Creates a new Identity Matrix
    static Mat<T> identity(int rows, int cols) {
        Mat<T> M(rows, cols); 
        launch_set_diagonal_kernel(M.d_data, rows, cols, 1.0f);
        M.cpu_dirty = true;
        return M;
    }

    void reset() {
      launch_reset_kernel(d_data, rows, cols, 0.0f);
      data.assign(rows * cols, 0.0f);
      cpu_dirty = true;
    }

    void assign(int r, int c) {
        rows = r; cols = c;
        
        // Only free GPU memory if it's NOT from the pool
        if (d_data && !from_pool) { 
          hipError_t err = hipFree(d_data);

          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "hipFree failed: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }
        }
        
        allocate_device_memory(); 
        reset();
        if (!from_pool) {
          hipError_t err = hipMemset(d_data, 0, r * c * sizeof(T));
          if (err != hipSuccess) {
            std::stringstream ss;
            ss << "hipMemset failed: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
          }
        }
    }

    void assign(int r, int c, std::vector<T> & data) {
      rows = r; cols = c;
      if (d_data && !from_pool) {
        hipError_t err = hipFree(d_data);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }
      allocate_device_memory();
      reset();
      hipError_t err = hipMemcpy(d_data, data.data(), r * c * sizeof(T), hipMemcpyHostToDevice);

      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }
      this->data = data;
      cpu_dirty = false;
    }

    // TODO: Fix this to work with the GPU Arena Memory.
    void clipGradients(float max_norm) {
      int size = rows * cols;
      if (size == 0) return;

      hipError_t err;

      // 1. Allocate device memory for the total sum
      float* d_total_sum_sq;
      float h_total_sum_sq = 0.0f;

      if (enable_arena) {
        d_total_sum_sq = (float*)global_arena.allocate(sizeof(float));
      } else {
        err = hipMalloc(&d_total_sum_sq, sizeof(float));
        if (err != hipSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate GPU Arena: " << hipGetErrorString(err);
            throw std::runtime_error(ss.str());
        }
      }

      // Initialize accumulator to 0
      err = hipMemset(d_total_sum_sq, 0, sizeof(float));
      if (err != hipSuccess) {
        std::stringstream ss;
        ss << "hipMemset failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
      }


      // 2. Launch Reduction Kernel
      int min_grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
      // Cap grid size to prevent overhead on huge vectors (1024 blocks is usually plenty to saturate)
      int grid_size = (min_grid_size > 1024) ? 1024 : min_grid_size;

      hipLaunchKernelGGL(kSumSquares, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0, 
          this->d_data, d_total_sum_sq, size);

      // 3. Copy sum back to Host to compute sqrt and check threshold
      // We must synchronize here to read the value
      err = hipMemcpy(&h_total_sum_sq, d_total_sum_sq, sizeof(float), hipMemcpyDeviceToHost);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }

      float total_norm = std::sqrt(h_total_sum_sq);

      // 4. Check logic and Launch Scaling Kernel if needed
      if (total_norm > max_norm) {
        float scale = max_norm / total_norm;

        // Launch scaling kernel
        hipLaunchKernelGGL(kApplyScale, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0, 
            this->d_data, scale, size);
      }

      if (!enable_arena) {
        err = hipFree(d_total_sum_sq);

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipFree failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
      }
    }

    void print(const std::string& label = "", int max_rows = 10, int max_cols = 10) {
      // 1. Ensure CPU data is synchronized with GPU
      to_cpu(); 

      std::cout << "\n--- Matrix Debug: " << label << " (" << rows << "x" << cols << ") ---" << std::endl;

      // 2. Calculate Basic Stats for rapid debugging
      float min_v = 1e9, max_v = -1e9, sum_v = 0, nan_count = 0;
      for (float v : data) {
        if (std::isnan(v)) nan_count++;
        else {
          if (v < min_v) min_v = v;
          if (v > max_v) max_v = v;
          sum_v += v;
        }
      }

      std::cout << "Stats -> Min: " << min_v << " | Max: " << max_v 
        << " | Avg: " << (rows * cols > 0 ? sum_v / (rows * cols) : 0) 
        << " | NaNs: " << nan_count << std::endl;

      if (nan_count > 0) {
        std::cout << "!!! WARNING: NaNs detected in matrix. Check learning rate or initialization." << std::endl;
      }

      // 3. Print the Grid
      int print_r = std::min(rows, max_rows);
      int print_c = std::min(cols, max_cols);

      for (int r = 0; r < print_r; ++r) {
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < print_c; ++c) {
          // Accessing data assuming row-major layout: row * cols + col
          float val = data[r * cols + c];
          printf("%8.4f ", val);
        }
        if (cols > max_cols) std::cout << "... (" << cols - max_cols << " more)";
        std::cout << std::endl;
      }

      if (rows > max_rows) {
        std::cout << "... (" << rows - max_rows << " more rows)" << std::endl;
      }
      std::cout << "-------------------------------------------\n" << std::endl;
    }
};


