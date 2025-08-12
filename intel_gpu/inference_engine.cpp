#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>

class IntelGPUInference {
private:
ov::Core core;
ov::CompiledModel compiled_model;
ov::InferRequest infer_request;
std::string model_path;
std::string device_name = “GPU”;

```
// OpenCL context for custom operations
cl_platform_id platform;
cl_device_id cl_device;
cl_context cl_context;
cl_command_queue cl_queue;
```

public:
IntelGPUInference(const std::string& model_path) : model_path(model_path) {
initializeOpenCL();
}

```
bool initializeOpenCL() {
    // Initialize OpenCL for custom GPU operations
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
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
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices == 0) {
        std::cerr << "No Intel GPU found for OpenCL\n";
        return false;
    }
    
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
    
    // Create OpenCL context and queue
    cl_context = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, nullptr);
    cl_queue = clCreateCommandQueue(cl_context, cl_device, 0, nullptr);
    
    return true;
}

bool loadModel() {
    try {
        // Load model
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        
        // Configure GPU device
        core.set_property(device_name, ov::cache_dir("./cache"));
        core.set_property(device_name, ov::inference_num_threads(4));
        
        // Enable GPU optimizations
        ov::AnyMap gpu_config = {
            {ov::hint::inference_precision.name(), ov::element::f16},
            {ov::hint::execution_mode.name(), ov::hint::ExecutionMode::PERFORMANCE},
            {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}
        };
        
        // Compile model for Intel GPU
        compiled_model = core.compile_model(model, device_name, gpu_config);
        infer_request = compiled_model.create_infer_request();
        
        std::cout << "Model loaded successfully on Intel GPU\n";
        std::cout << "Input shape: ";
        auto input_tensor = infer_request.get_input_tensor();
        auto input_shape = input_tensor.get_shape();
        for (size_t dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> preprocessImage(const cv::Mat& image, int target_width = 224, int target_height = 224) {
    cv::Mat resized, normalized;
    
    // Resize image
    cv::resize(image, resized, cv::Size(target_width, target_height));
    
    // Convert to float and normalize
    resized.convertTo(normalized, CV_32F, 1.0/255.0);
    
    // Convert BGR to RGB
    cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
    
    // Convert to CHW format (Channels, Height, Width)
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    
    std::vector<float> input_data;
    input_data.reserve(3 * target_height * target_width);
    
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = channels[c];
        input_data.insert(input_data.end(), 
                        (float*)channel.datastart, 
                        (float*)channel.dataend);
    }
    
    return input_data;
}

std::vector<float> inferImage(const cv::Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Preprocess image
    auto input_data = preprocessImage(image);
    
    // Set input tensor
    auto input_tensor = infer_request.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), input_data.data(), 
               input_data.size() * sizeof(float));
    
    // Run inference
    infer_request.infer();
    
    // Get output
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    
    size_t output_size = 1;
    for (size_t dim : output_shape) {
        output_size *= dim;
    }
    
    std::vector<float> output_data(output_size);
    std::memcpy(output_data.data(), output_tensor.data<float>(), 
               output_size * sizeof(float));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    
    return output_data;
}

// Custom GPU kernel for post-processing
std::vector<float> gpuSoftmax(const std::vector<float>& logits) {
    size_t data_size = logits.size() * sizeof(float);
    
    // Create OpenCL buffers
    cl_mem input_buf = clCreateBuffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     data_size, (void*)logits.data(), nullptr);
    cl_mem output_buf = clCreateBuffer(cl_context, CL_MEM_WRITE_ONLY, data_size, nullptr, nullptr);
    
    // Softmax kernel
    const char* kernel_source = R"(
        __kernel void softmax(__global const float* input, __global float* output, int size) {
            int gid = get_global_id(0);
            
            if (gid == 0) {
                // Find max for numerical stability
                float max_val = input[0];
                for (int i = 1; i < size; i++) {
                    max_val = fmax(max_val, input[i]);
                }
                
                // Calculate sum of exponentials
                float sum = 0.0f;
                for (int i = 0; i < size; i++) {
                    sum += exp(input[i] - max_val);
                }
                
                // Calculate softmax
                for (int i = 0; i < size; i++) {
                    output[i] = exp(input[i] - max_val) / sum;
                }
            }
        }
    )";
    
    cl_program program = clCreateProgramWithSource(cl_context, 1, &kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &cl_device, nullptr, nullptr, nullptr);
    
    cl_kernel kernel = clCreateKernel(program, "softmax", nullptr);
    
    int size = static_cast<int>(logits.size());
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 2, sizeof(int), &size);
    
    size_t global_size = 1;
    clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    
    std::vector<float> result(logits.size());
    clEnqueueReadBuffer(cl_queue, output_buf, CL_TRUE, 0, data_size, result.data(), 0, nullptr, nullptr);
    
    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return result;
}

// Non-Maximum Suppression on GPU
std::vector<Detection> gpuNMS(const std::vector<Detection>& detections, float iou_threshold = 0.5f) {
    if (detections.empty()) return {};
    
    size_t num_detections = detections.size();
    size_t data_size = num_detections * sizeof(Detection);
    
    cl_mem detections_buf = clCreateBuffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          data_size, (void*)detections.data(), nullptr);
    cl_mem keep_buf = clCreateBuffer(cl_context, CL_MEM_WRITE_ONLY, 
                                    num_detections * sizeof(int), nullptr, nullptr);
    
    const char* nms_kernel = R"(
        typedef struct {
            float x, y, width, height;
            float confidence;
            int class_id;
        } Detection;
        
        float calculateIoU(Detection a, Detection b) {
            float x1 = fmax(a.x, b.x);
            float y1 = fmax(a.y, b.y);
            float x2 = fmin(a.x + a.width, b.x + b.width);
            float y2 = fmin(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) return 0.0f;
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area_a = a.width * a.height;
            float area_b = b.width * b.height;
            float union_area = area_a + area_b - intersection;
            
            return intersection / union_area;
        }
        
        __kernel void nms(__global const Detection* detections, __global int* keep, 
                        int num_detections, float iou_threshold) {
            int idx = get_global_id(0);
            
            if (idx >= num_detections) return;
            
            keep[idx] = 1;  // Initially keep all
            
            // Check against all higher confidence detections
            for (int i = 0; i < idx; i++) {
                if (keep[i] && detections[i].class_id == detections[idx].class_id) {
                    float iou = calculateIoU(detections[i], detections[idx]);
                    if (iou > iou_threshold) {
                        keep[idx] = 0;  // Suppress this detection
                        break;
                    }
                }
            }
        }
    )";
    
    cl_program program = clCreateProgramWithSource(cl_context, 1, &nms_kernel, nullptr, nullptr);
    clBuildProgram(program, 1, &cl_device, nullptr, nullptr, nullptr);
    
    cl_kernel kernel = clCreateKernel(program, "nms", nullptr);
    
    int num_det = static_cast<int>(num_detections);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &detections_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &keep_buf);
    clSetKernelArg(kernel, 2, sizeof(int), &num_det);
    clSetKernelArg(kernel, 3, sizeof(float), &iou_threshold);
    
    size_t global_size = num_detections;
    clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    
    std::vector<int> keep(num_detections);
    clEnqueueReadBuffer(cl_queue, keep_buf, CL_TRUE, 0, 
                       num_detections * sizeof(int), keep.data(), 0, nullptr, nullptr);
    
    // Filter kept detections
    std::vector<Detection> filtered;
    for (size_t i = 0; i < num_detections; i++) {
        if (keep[i]) {
            filtered.push_back(detections[i]);
        }
    }
    
    // Cleanup
    clReleaseMemObject(detections_buf);
    clReleaseMemObject(keep_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return filtered;
}

struct Detection {
    float x, y, width, height;
    float confidence;
    int class_id;
};

std::vector<Detection> parseYOLOOutput(const std::vector<float>& output, 
                                      float conf_threshold = 0.5f,
                                      int img_width = 640, int img_height = 640) {
    std::vector<Detection> detections;
    
    // YOLO output format: [batch, num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    int num_detections = output.size() / 85;
    
    for (int i = 0; i < num_detections; i++) {
        int offset = i * 85;
        
        float confidence = output[offset + 4];
        if (confidence < conf_threshold) continue;
        
        // Find class with highest probability
        int best_class = 0;
        float best_prob = output[offset + 5];
        for (int c = 1; c < 80; c++) {
            float prob = output[offset + 5 + c];
            if (prob > best_prob) {
                best_prob = prob;
                best_class = c;
            }
        }
        
        float final_conf = confidence * best_prob;
        if (final_conf < conf_threshold) continue;
        
        // Convert from center format to top-left format
        float cx = output[offset] * img_width;
        float cy = output[offset + 1] * img_height;
        float w = output[offset + 2] * img_width;
        float h = output[offset + 3] * img_height;
        
        Detection det;
        det.x = cx - w / 2;
        det.y = cy - h / 2;
        det.width = w;
        det.height = h;
        det.confidence = final_conf;
        det.class_id = best_class;
        
        detections.push_back(det);
    }
    
    return detections;
}

void benchmarkPerformance(const cv::Mat& test_image, int num_iterations = 100) {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    std::vector<double> inference_times;
    
    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = inferImage(test_image);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        inference_times.push_back(duration.count() / 1000.0); // Convert to ms
    }
    
    // Calculate statistics
    double sum = 0;
    for (double time : inference_times) {
        sum += time;
    }
    double avg_time = sum / num_iterations;
    
    std::sort(inference_times.begin(), inference_times.end());
    double median_time = inference_times[num_iterations / 2];
    double min_time = inference_times[0];
    double max_time = inference_times[num_iterations - 1];
    
    std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
    std::cout << "Median inference time: " << median_time << " ms" << std::endl;
    std::cout << "Min inference time: " << min_time << " ms" << std::endl;
    std::cout << "Max inference time: " << max_time << " ms" << std::endl;
    std::cout << "FPS (avg): " << 1000.0 / avg_time << std::endl;
}

~IntelGPUInference() {
    if (cl_queue) clReleaseCommandQueue(cl_queue);
    if (cl_context) clReleaseContext(cl_context);
}
```

};

// Example usage
int main(int argc, char* argv[]) {
if (argc < 3) {
std::cout << “Usage: “ << argv[0] << “ <model_path> <image_path>” << std::endl;
return -1;
}

```
std::string model_path = argv[1];
std::string image_path = argv[2];

// Initialize inference engine
IntelGPUInference inference(model_path);

if (!inference.loadModel()) {
    std::cerr << "Failed to load model" << std::endl;
    return -1;
}

// Load test image
cv::Mat image = cv::imread(image_path);
if (image.empty()) {
    std::cerr << "Failed to load image" << std::endl;
    return -1;
}

std::cout << "Running inference on image: " << image_path << std::endl;

// Run inference
auto result = inference.inferImage(image);

// Apply GPU-accelerated softmax for classification
auto probabilities = inference.gpuSoftmax(result);

// Find top predictions
std::vector<std::pair<int, float>> indexed_probs;
for (int i = 0; i < probabilities.size(); i++) {
    indexed_probs.push_back({i, probabilities[i]});
}

std::sort(indexed_probs.begin(), indexed_probs.end(), 
          [](const auto& a, const auto& b) { return a.second > b.second; });

std::cout << "\nTop 5 predictions:" << std::endl;
for (int i = 0; i < std::min(5, (int)indexed_probs.size()); i++) {
    std::cout << "Class " << indexed_probs[i].first 
              << ": " << indexed_probs[i].second * 100 << "%" << std::endl;
}

// Performance benchmark
inference.benchmarkPerformance(image);

return 0;
```

}