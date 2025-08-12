#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

class IntelGPUCompute {
private:
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

public:
bool initialize() {
// Get Intel GPU platform
cl_uint numPlatforms;
clGetPlatformIDs(0, nullptr, &numPlatforms);

```
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    // Find Intel platform
    for (auto p : platforms) {
        char vendor[256];
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        if (std::string(vendor).find("Intel") != std::string::npos) {
            platform = p;
            break;
        }
    }
    
    // Get Intel GPU device
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices == 0) {
        std::cerr << "No Intel GPU found\n";
        return false;
    }
    
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
    // Create context and command queue
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    return true;
}

void vectorAdd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    size_t dataSize = a.size() * sizeof(float);
    
    // Create buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 dataSize, (void*)a.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 dataSize, (void*)b.data(), nullptr);
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, nullptr, nullptr);
    
    // Kernel source
    const char* kernelSource = R"(
        __kernel void vector_add(__global const float* a, __global const float* b, __global float* result) {
            int gid = get_global_id(0);
            result[gid] = a[gid] + b[gid];
        }
    )";
    
    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);
    
    // Set arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
    
    // Execute kernel
    size_t globalSize = a.size();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    // Read result
    result.resize(a.size());
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, dataSize, result.data(), 0, nullptr, nullptr);
    
    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void matrixMultiply(const std::vector<float>& matA, const std::vector<float>& matB, 
                   std::vector<float>& matC, int width) {
    size_t dataSize = matA.size() * sizeof(float);
    
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 dataSize, (void*)matA.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 dataSize, (void*)matB.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, nullptr, nullptr);
    
    const char* kernelSource = R"(
        __kernel void matrix_multiply(__global const float* A, __global const float* B, 
                                    __global float* C, int width) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    )";
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    
    size_t globalSize[2] = {(size_t)width, (size_t)width};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    
    matC.resize(matA.size());
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, dataSize, matC.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

~IntelGPUCompute() {
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}
```

};

// Example usage
int main() {
IntelGPUCompute gpu;
if (!gpu.initialize()) {
return -1;
}

```
// Vector addition example
std::vector<float> a = {1, 2, 3, 4, 5};
std::vector<float> b = {5, 4, 3, 2, 1};
std::vector<float> result;

gpu.vectorAdd(a, b, result);

std::cout << "Vector addition result: ";
for (float val : result) {
    std::cout << val << " ";
}
std::cout << std::endl;

return 0;
```

}