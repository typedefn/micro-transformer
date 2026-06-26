#pragma once
#include "Common.hpp"

// ----------------------------------------------------------------------
// CUSTOM HIP KERNELS FOR ROW-MAJOR MATRIX MULTIPLICATION
// ----------------------------------------------------------------------

template <typename T>
__global__ void hip_batched_mul(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < M && col < N) {
        const T* a_batch = A + batch * stride_a;
        const T* b_batch = B + batch * stride_b;
        T* c_batch = C + batch * stride_c;

        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a_batch[row * K + k] * b_batch[k * N + col];
        }
        c_batch[row * N + col] = sum;
    }
}

template <typename T>
__global__ void hip_batched_mul_transpose(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < M && col < N) {
        const T* a_batch = A + batch * stride_a;
        const T* b_batch = B + batch * stride_b;
        T* c_batch = C + batch * stride_c;

        T sum = 0;
        for (int k = 0; k < K; ++k) {
            // B is transposed (B^T). We read it as B[col, k] instead of B[k, col]
            sum += a_batch[row * K + k] * b_batch[col * K + k];
        }
        c_batch[row * N + col] = sum;
    }
}

template <typename T>
__global__ void hip_batched_mul_lhs_transpose(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < M && col < N) {
        const T* a_batch = A + batch * stride_a;
        const T* b_batch = B + batch * stride_b;
        T* c_batch = C + batch * stride_c;

        T sum = 0;
        for (int k = 0; k < K; ++k) {
            // A is transposed (A^T). We read it as A[k, row] instead of A[row, k]
            sum += a_batch[k * M + row] * b_batch[k * N + col];
        }
        c_batch[row * N + col] = sum;
    }
}

template <typename T>
__global__ void hip_mul(
    const T* A, const T* B, T* C,
    int M, int N, int K) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void kComputeLogitGradients(const float* predictions, const float* targets, float* gradients, int size);
__global__ void kSoftCrossEntropy(const float* predicted, const float* targets, float* total_loss, int size, float epsilon);
// One thread per element in the final concatenated matrix
__global__ void kConcatenateHeads(float** head_ptrs, float* dest, int num_heads, int head_dim, int total_tokens);
// --- LAYER NORM KERNELS ---
// 1. Compute Mean and Variance (Fused Kernel)
// One block per row (sequence element). Threads reduce locally.
__global__ void kLayerNormStats(const float* src, float* mean, float* variance, int rows, int cols);
// 2. Forward Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void kLayerNormForward(const float* src, float* dest, const float* mean, const float* var, 
                                  const float* gamma, const float* beta, 
                                  int rows, int cols, float epsilon);
// 3. Backward Pass (Fused Gradient Calculation)
// Computes dGamma, dBeta, and dInput in one go is hard, 
// so we split: Step A (Accumulate Params), Step B (Compute dInput)
__global__ void kLayerNormBackwardParams(const float* d_out, const float* src, const float* mean, const float* var,
                                         float* d_gamma, float* d_beta, 
                                         int rows, int cols, float epsilon);
    // We parallelize over COLUMNS (Features). 
__global__ void kLayerNormBackwardInput(const float* d_out, const float* src, const float* mean, const float* var,
                                        const float* gamma, float* d_in,
                                        int rows, int cols, float epsilon);
// Compute Max per row (for numerical stability)
__global__ void kRowMax(const float* src, float* max_vals, int rows, int cols);
// Compute Sum of Exponentials per row
__global__ void kRowSumExp(const float* src, const float* max_vals, float* sum_vals, int rows, int cols);
// Final Softmax: Exp(x - max) / Sum
// Fixed: Handles cases where max_val is -Infinity or sum is 0 to prevent NaNs
__global__ void kApplySoftmax(const float* src, float* dest, const float* max_vals, const float* sum_vals, int rows, int cols);
// Softmax Backward: d_in = softmax * (d_out - sum(d_out * softmax))
// This kernel calculates the dot product (sum(d_out * softmax)) per row
__global__ void kSoftmaxGradDot(const float* grad_output, const float* softmax_output, float* dot_products, int rows, int cols);
// Softmax Backward Final Calculation
__global__ void kApplySoftmaxBackward(const float* grad_output, const float* softmax_output, const float* dot_products, float* grad_input, int rows, int cols);
// Sums columns: Collapses an (rows, cols) matrix into a (1, cols) vector
// Used for Bias Gradients in vectorized layers
__global__ void kSumColumns(const float* src, float* dest, int rows, int cols);
// Inside Kernel 1 (Element-wise Addition) - Update to this:
__global__ void kAddBroadcast(const float* a, const float* b, float* c, int rows, int cols, int b_rows);
// -------------------------------------------------------------------------
// Kernel: Parallel Sum of Squares (Reduction)
// -------------------------------------------------------------------------
__global__ void kSumSquares(const float* __restrict__ gradients, float* total_sum_sq, int n);
// Splits a large gradient matrix into smaller head-specific matrices
__global__ void kSplitGradients(const float* src, float** head_grad_ptrs, int num_heads, int head_dim, int total_tokens);
// -------------------------------------------------------------------------
// Kernel: Apply Scaling
// -------------------------------------------------------------------------
__global__ void kApplyScale(float* gradients, float scale, int n);
__global__ void kSetDiagonal(float* out, int rows, int cols, float val);
__global__ void kReset(float* out, int rows, int cols, float val);
__global__ void kDropout(float* out, int size, float p, unsigned int seed, int step);
__global__ void kGelu(const float* in, float* out, int size);
__global__ void kGeluBackward(const float* inputs, const float* gradients, float* out, int size);
// Element-wise Addition
__global__ void kAdd(const float* a, const float* b, float* c, int size);
// Matrix Transpose
__global__ void kTranspose(const float* in, float* out, int rows, int cols);
// Banded Matrix Multiplication
__global__ void kBandedMul(const float* A, const float* B, float* C,
                           int rows, int cols, int common, int window);
// Corrected kScaleMask for Batched Stride (Tall & Narrow Matrix)
__global__ void kScaleMask(const float* in, float* out, int rows, int cols, float scale, int valid_seq_len, int seq_len);
__global__ void kScale(const float* in, float* out, int size, float scale);
// Element-wise Product
__global__ void kElementWiseMul(const float* a, const float* b, float* c, int size);

class GPUMemoryArena {
    void* base_ptr = nullptr;
    size_t total_size = 0;
    // Use atomic for lock-free thread safety
    std::atomic<size_t> offset{0}; 

public:
    void init(size_t size_bytes = 1024 * 1024 * 512) {
        if (base_ptr) return; 
        total_size = size_bytes;
        hipError_t err = (hipMalloc(&base_ptr, total_size));
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
            total_size = 0; // Reset state entirely
            offset.store(0, std::memory_order_relaxed);
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
void launch_set_diagonal_kernel(float* data, int rows, int cols, float val);
void launch_reset_kernel(float* data, int rows, int cols, float val);

// Global instance
inline GPUMemoryArena global_arena;
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
    Mat() = default;
    Mat(int rows, int cols) : rows(rows), cols(cols) {
        from_pool = false; 
        allocate_device_memory();
        reset();
        cpu_dirty = false;
    }

    Mat(int rows, int cols, const std::vector<T> & input_data) : rows(rows), cols(cols) {
        from_pool = false; 
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
    void batched_mul(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
      int M = rows / batch_count;     // Rows of A
      int K = cols;                   // Cols of A / Rows of B
      int N = rhs.cols;               // Cols of B
 
      dim3 block(16, 16);
      dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
 
      hip_batched_mul<<<grid, block>>>(
          this->d_data, rhs.d_data, result.d_data,
          M, N, K,
          stride_a, stride_b, stride_c
      );
      
      result.cpu_dirty = true;
    }

    // C = A * B^T (Batched)
    // Used for: Q * K^T
    void batched_mul_transpose(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
      int M = rows / batch_count;     // Rows of A
      int K = cols;                   // Cols of A
      int N = rhs.rows / batch_count; // Rows of B (becomes cols of B^T)
 
      dim3 block(16, 16);
      dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
 
      hip_batched_mul_transpose<<<grid, block>>>(
          this->d_data, rhs.d_data, result.d_data,
          M, N, K,
          stride_a, stride_b, stride_c
      );
      
      result.cpu_dirty = true;
    }

    // C = A^T * B (Batched)
    // Used for: dV = S^T * dO, dK = dS^T * Q
    void batched_mul_lhs_transpose(const Mat<T>& rhs, Mat<T>& result, int batch_count, int stride_a, int stride_b, int stride_c) {
      int K = rows / batch_count;    // Rows of A (becomes inner dim K)
      int M = cols;                  // Cols of A (becomes rows of A^T)
      int N = rhs.cols;              // Cols of B

      dim3 block(16, 16);
      dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);

      hip_batched_mul_lhs_transpose<<<grid, block>>>(
          this->d_data, rhs.d_data, result.d_data,
          M, N, K,
          stride_a, stride_b, stride_c
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
        hipError_t err = (hipMalloc(&d_loss_ptr, sizeof(float)));
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
        hipError_t err = (hipMalloc(&d_head_ptrs, num_heads * sizeof(T*)));
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

   Mat& operator=(Mat&& other) noexcept {
      if (this != &other) {
        if (d_data && !from_pool) {
          hipFree(d_data);
        }   
        rows = other.rows;
        cols = other.cols;
        d_data = other.d_data;
        data = std::move(other.data);
        cpu_dirty = other.cpu_dirty;
        from_pool = other.from_pool;

        other.d_data = nullptr; // Prevent double free
        other.rows = 0;
        other.cols = 0;
      }   
      return *this;
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
        if (size == 0) { 
          d_data = nullptr;
          std::stringstream warn;
          warn << "Requested 0 bytes during allocate device memory.  rows = " << rows << " cols = " << cols << std::endl;
          std::cerr << warn.str();
        }
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
      // Standard unbatched matrix multiplication
      assert(cols == rhs.rows);

      if (result.rows != rows || result.cols != rhs.cols) {
        throw std::runtime_error("MUL - invalid size");
      }

      int M = rows;
      int K = cols;
      int N = rhs.cols;

      dim3 block(16, 16);
      dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);

      hip_mul<<<grid, block>>>(
          this->d_data, rhs.d_data, result.d_data,
          M, N, K
          );

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


