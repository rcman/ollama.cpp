import pyopencl as cl
import numpy as np
import time

class IntelGPUPython:
def **init**(self):
self.context = None
self.queue = None
self.device = None

```
def initialize(self):
    """Initialize Intel GPU context"""
    try:
        # Get all platforms
        platforms = cl.get_platforms()
        
        # Find Intel platform
        intel_platform = None
        for platform in platforms:
            if 'intel' in platform.vendor.lower():
                intel_platform = platform
                break
        
        if not intel_platform:
            print("No Intel platform found")
            return False
        
        # Get Intel GPU devices
        devices = intel_platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            print("No Intel GPU devices found")
            return False
        
        self.device = devices[0]
        print(f"Using device: {self.device.name}")
        print(f"Max compute units: {self.device.max_compute_units}")
        print(f"Global memory: {self.device.global_mem_size // (1024**3)} GB")
        
        # Create context and command queue
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False

def vector_operations(self, a, b):
    """Perform vector addition and multiplication on GPU"""
    # Convert to numpy arrays
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    
    # Create result arrays
    add_result = np.empty_like(a_np)
    mul_result = np.empty_like(a_np)
    
    # Create buffers
    a_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
    b_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
    add_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, add_result.nbytes)
    mul_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, mul_result.nbytes)
    
    # OpenCL kernel
    kernel_code = """
    __kernel void vector_add(__global const float* a, __global const float* b, __global float* result) {
        int gid = get_global_id(0);
        result[gid] = a[gid] + b[gid];
    }
    
    __kernel void vector_multiply(__global const float* a, __global const float* b, __global float* result) {
        int gid = get_global_id(0);
        result[gid] = a[gid] * b[gid];
    }
    """
    
    # Build program
    program = cl.Program(self.context, kernel_code).build()
    
    # Execute kernels
    program.vector_add(self.queue, a_np.shape, None, a_buf, b_buf, add_buf)
    program.vector_multiply(self.queue, a_np.shape, None, a_buf, b_buf, mul_buf)
    
    # Read results
    cl.enqueue_copy(self.queue, add_result, add_buf)
    cl.enqueue_copy(self.queue, mul_result, mul_buf)
    
    return add_result.tolist(), mul_result.tolist()

def matrix_multiply(self, A, B):
    """GPU matrix multiplication"""
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)
    
    assert A_np.shape[1] == B_np.shape[0], "Matrix dimensions don't match"
    
    M, K = A_np.shape
    K2, N = B_np.shape
    
    C_np = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    A_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A_np)
    B_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B_np)
    C_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, C_np.nbytes)
    
    kernel_code = """
    __kernel void matrix_multiply(__global const float* A, __global const float* B, 
                                __global float* C, int M, int N, int K) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    
    program = cl.Program(self.context, kernel_code).build()
    
    # Execute kernel
    global_size = (M, N)
    program.matrix_multiply(self.queue, global_size, None, 
                          A_buf, B_buf, C_buf, 
                          np.int32(M), np.int32(N), np.int32(K))
    
    # Read result
    cl.enqueue_copy(self.queue, C_np, C_buf)
    
    return C_np.tolist()

def image_processing(self, image_data, width, height):
    """GPU-accelerated image processing example"""
    img_np = np.array(image_data, dtype=np.float32)
    result_np = np.zeros_like(img_np)
    
    img_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img_np)
    result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result_np.nbytes)
    
    # Simple blur kernel
    kernel_code = """
    __kernel void blur_filter(__global const float* input, __global float* output, 
                            int width, int height) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x >= width || y >= height) return;
        
        int idx = y * width + x;
        float sum = 0.0f;
        int count = 0;
        
        // 3x3 blur
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
    """
    
    program = cl.Program(self.context, kernel_code).build()
    
    global_size = (width, height)
    program.blur_filter(self.queue, global_size, None, 
                      img_buf, result_buf, 
                      np.int32(width), np.int32(height))
    
    cl.enqueue_copy(self.queue, result_np, result_buf)
    
    return result_np.tolist()

def benchmark_performance(self, size=1000000):
    """Benchmark GPU vs CPU performance"""
    # Generate test data
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # CPU timing
    start_time = time.time()
    cpu_result = a + b
    cpu_time = time.time() - start_time
    
    # GPU timing
    start_time = time.time()
    gpu_add, _ = self.vector_operations(a.tolist(), b.tolist())
    gpu_time = time.time() - start_time
    
    print(f"Vector size: {size}")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    return cpu_time, gpu_time
```

def main():
# Initialize Intel GPU
gpu = IntelGPUPython()
if not gpu.initialize():
print(“Failed to initialize Intel GPU”)
return

```
# Vector operations example
print("=== Vector Operations ===")
a = [1, 2, 3, 4, 5]
b = [5, 4, 3, 2, 1]
add_result, mul_result = gpu.vector_operations(a, b)
print(f"A + B = {add_result}")
print(f"A * B = {mul_result}")

# Matrix multiplication example
print("\n=== Matrix Multiplication ===")
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = gpu.matrix_multiply(A, B)
print(f"Matrix A: {A}")
print(f"Matrix B: {B}")
print(f"A * B = {C}")

# Performance benchmark
print("\n=== Performance Benchmark ===")
gpu.benchmark_performance()

# Image processing example
print("\n=== Image Processing ===")
# Create a simple test image (random noise)
width, height = 100, 100
test_image = np.random.rand(height * width).tolist()
blurred = gpu.image_processing(test_image, width, height)
print(f"Processed {width}x{height} image")
```

if **name** == “**main**”:
main()