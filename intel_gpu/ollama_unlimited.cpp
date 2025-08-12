#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// Configuration constants
constexpr size_t CHUNK_SIZE = 1024 * 1024 * 512; // 512MB chunks
constexpr size_t MAX_MEMORY_CACHE = 1024 * 1024 * 1024 * 8ULL; // 8GB cache
constexpr size_t PREFETCH_CHUNKS = 4; // Number of chunks to prefetch

// Forward declarations
class MemoryPool;
class StreamingTensor;
class ChunkedModel;
class InferenceEngine;

// Memory management system
class MemoryPool {
private:
struct MemoryBlock {
void* ptr;
size_t size;
bool in_use;
uint64_t last_access;

```
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false), last_access(0) {}
};

std::vector<std::unique_ptr<MemoryBlock>> blocks;
std::mutex pool_mutex;
size_t total_allocated;
size_t max_pool_size;
```

public:
MemoryPool(size_t max_size = MAX_MEMORY_CACHE)
: total_allocated(0), max_pool_size(max_size) {}

```
~MemoryPool() {
    cleanup();
}

void* allocate(size_t size, bool use_mmap = false) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Try to find existing free block
    for (auto& block : blocks) {
        if (!block->in_use && block->size >= size) {
            block->in_use = true;
            block->last_access = getCurrentTime();
            return block->ptr;
        }
    }
    
    // Check if we need to free some memory
    if (total_allocated + size > max_pool_size) {
        evictLRU(size);
    }
    
    void* ptr = nullptr;
    if (use_mmap && size >= 1024 * 1024) { // Use mmap for large allocations
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            ptr = nullptr;
        }
    } else {
        ptr = aligned_alloc(64, size); // 64-byte alignment for SIMD
    }
    
    if (ptr) {
        blocks.push_back(std::make_unique<MemoryBlock>(ptr, size));
        blocks.back()->in_use = true;
        blocks.back()->last_access = getCurrentTime();
        total_allocated += size;
    }
    
    return ptr;
}

void deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& block : blocks) {
        if (block->ptr == ptr) {
            block->in_use = false;
            break;
        }
    }
}

void cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& block : blocks) {
        if (block->size >= 1024 * 1024) {
            munmap(block->ptr, block->size);
        } else {
            free(block->ptr);
        }
    }
    blocks.clear();
    total_allocated = 0;
}
```

private:
uint64_t getCurrentTime() {
return std::chrono::steady_clock::now().time_since_epoch().count();
}

```
void evictLRU(size_t needed_size) {
    // Sort by last access time and evict unused blocks
    std::vector<std::unique_ptr<MemoryBlock>*> unused_blocks;
    
    for (auto& block : blocks) {
        if (!block->in_use) {
            unused_blocks.push_back(&block);
        }
    }
    
    std::sort(unused_blocks.begin(), unused_blocks.end(),
             [](const auto& a, const auto& b) {
                 return (*a)->last_access < (*b)->last_access;
             });
    
    size_t freed = 0;
    for (auto& block_ptr : unused_blocks) {
        auto& block = *block_ptr;
        
        if (block->size >= 1024 * 1024) {
            munmap(block->ptr, block->size);
        } else {
            free(block->ptr);
        }
        
        freed += block->size;
        total_allocated -= block->size;
        
        // Remove from blocks vector
        blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                    [&block](const auto& b) { return b.get() == block.get(); }),
                    blocks.end());
        
        if (freed >= needed_size) break;
    }
}
```

};

// Streaming tensor class for handling large tensors
class StreamingTensor {
private:
struct TensorChunk {
void* data;
size_t size;
size_t offset;
bool loaded;
std::mutex chunk_mutex;

```
    TensorChunk(size_t s, size_t o) : data(nullptr), size(s), offset(o), loaded(false) {}
};

std::vector<std::unique_ptr<TensorChunk>> chunks;
std::string file_path;
int file_descriptor;
size_t total_size;
size_t chunk_size;
MemoryPool* memory_pool;

// Prefetching system
std::thread prefetch_thread;
std::queue<size_t> prefetch_queue;
std::mutex prefetch_mutex;
std::condition_variable prefetch_cv;
bool stop_prefetch;
```

public:
StreamingTensor(const std::string& path, size_t total_sz, MemoryPool* pool)
: file_path(path), total_size(total_sz), chunk_size(CHUNK_SIZE),
memory_pool(pool), stop_prefetch(false) {

```
    file_descriptor = open(file_path.c_str(), O_RDONLY);
    if (file_descriptor == -1) {
        throw std::runtime_error("Failed to open tensor file: " + file_path);
    }
    
    // Create chunks
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    chunks.reserve(num_chunks);
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        size_t size = std::min(chunk_size, total_size - offset);
        chunks.push_back(std::make_unique<TensorChunk>(size, offset));
    }
    
    // Start prefetch thread
    prefetch_thread = std::thread(&StreamingTensor::prefetchWorker, this);
}

~StreamingTensor() {
    stop_prefetch = true;
    prefetch_cv.notify_all();
    
    if (prefetch_thread.joinable()) {
        prefetch_thread.join();
    }
    
    if (file_descriptor != -1) {
        close(file_descriptor);
    }
    
    // Free all loaded chunks
    for (auto& chunk : chunks) {
        if (chunk->loaded && chunk->data) {
            memory_pool->deallocate(chunk->data);
        }
    }
}

void* getChunk(size_t chunk_index) {
    if (chunk_index >= chunks.size()) {
        return nullptr;
    }
    
    auto& chunk = chunks[chunk_index];
    std::lock_guard<std::mutex> lock(chunk->chunk_mutex);
    
    if (!chunk->loaded) {
        loadChunk(chunk_index);
    }
    
    // Prefetch next chunks
    schedulePrefetch(chunk_index + 1, PREFETCH_CHUNKS);
    
    return chunk->data;
}

size_t getChunkSize(size_t chunk_index) const {
    if (chunk_index >= chunks.size()) {
        return 0;
    }
    return chunks[chunk_index]->size;
}

size_t getNumChunks() const {
    return chunks.size();
}

size_t getTotalSize() const {
    return total_size;
}
```

private:
void loadChunk(size_t chunk_index) {
auto& chunk = chunks[chunk_index];

```
    if (chunk->loaded) return;
    
    chunk->data = memory_pool->allocate(chunk->size);
    if (!chunk->data) {
        throw std::runtime_error("Failed to allocate memory for tensor chunk");
    }
    
    // Read from file
    ssize_t bytes_read = pread(file_descriptor, chunk->data, chunk->size, chunk->offset);
    if (bytes_read != static_cast<ssize_t>(chunk->size)) {
        memory_pool->deallocate(chunk->data);
        chunk->data = nullptr;
        throw std::runtime_error("Failed to read tensor chunk from file");
    }
    
    chunk->loaded = true;
}

void schedulePrefetch(size_t start_chunk, size_t count) {
    std::lock_guard<std::mutex> lock(prefetch_mutex);
    
    for (size_t i = start_chunk; i < std::min(start_chunk + count, chunks.size()); ++i) {
        if (!chunks[i]->loaded) {
            prefetch_queue.push(i);
        }
    }
    
    prefetch_cv.notify_one();
}

void prefetchWorker() {
    while (!stop_prefetch) {
        std::unique_lock<std::mutex> lock(prefetch_mutex);
        prefetch_cv.wait(lock, [this] { return !prefetch_queue.empty() || stop_prefetch; });
        
        if (stop_prefetch) break;
        
        size_t chunk_index = prefetch_queue.front();
        prefetch_queue.pop();
        lock.unlock();
        
        try {
            auto& chunk = chunks[chunk_index];
            std::lock_guard<std::mutex> chunk_lock(chunk->chunk_mutex);
            
            if (!chunk->loaded) {
                loadChunk(chunk_index);
            }
        } catch (const std::exception& e) {
            // Prefetch failure is not critical, just log
            std::cerr << "Prefetch failed for chunk " << chunk_index 
                      << ": " << e.what() << std::endl;
        }
    }
}
```

};

// Chunked model class for handling large models
class ChunkedModel {
private:
struct LayerInfo {
std::string name;
std::vector<size_t> shape;
size_t data_size;
std::string file_path;
std::unique_ptr<StreamingTensor> tensor;
};

```
std::vector<LayerInfo> layers;
MemoryPool memory_pool;
std::string model_directory;
```

public:
ChunkedModel(const std::string& model_dir) : model_directory(model_dir) {}

```
bool loadModel(const std::string& manifest_path) {
    std::ifstream manifest(manifest_path);
    if (!manifest.is_open()) {
        std::cerr << "Failed to open model manifest: " << manifest_path << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        LayerInfo layer;
        if (parseLayerInfo(line, layer)) {
            // Create streaming tensor for this layer
            std::string full_path = model_directory + "/" + layer.file_path;
            try {
                layer.tensor = std::make_unique<StreamingTensor>(
                    full_path, layer.data_size, &memory_pool);
                layers.push_back(std::move(layer));
            } catch (const std::exception& e) {
                std::cerr << "Failed to load layer " << layer.name 
                          << ": " << e.what() << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "Loaded " << layers.size() << " layers successfully" << std::endl;
    return !layers.empty();
}

StreamingTensor* getLayer(const std::string& layer_name) {
    for (auto& layer : layers) {
        if (layer.name == layer_name) {
            return layer.tensor.get();
        }
    }
    return nullptr;
}

size_t getLayerCount() const {
    return layers.size();
}

const LayerInfo& getLayerInfo(size_t index) const {
    return layers[index];
}

void printModelInfo() const {
    std::cout << "Model Information:" << std::endl;
    std::cout << "=================" << std::endl;
    
    size_t total_size = 0;
    for (const auto& layer : layers) {
        std::cout << "Layer: " << layer.name 
                  << ", Size: " << (layer.data_size / (1024 * 1024)) << " MB"
                  << ", Shape: [";
        
        for (size_t i = 0; i < layer.shape.size(); ++i) {
            std::cout << layer.shape[i];
            if (i < layer.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        total_size += layer.data_size;
    }
    
    std::cout << "Total model size: " << (total_size / (1024 * 1024 * 1024)) 
              << " GB" << std::endl;
}
```

private:
bool parseLayerInfo(const std::string& line, LayerInfo& layer) {
// Simple format: name,shape1:shape2:shape3,size_bytes,file_path
std::istringstream iss(line);
std::string token;

```
    // Parse name
    if (!std::getline(iss, token, ',')) return false;
    layer.name = token;
    
    // Parse shape
    if (!std::getline(iss, token, ',')) return false;
    std::istringstream shape_stream(token);
    std::string dim;
    while (std::getline(shape_stream, dim, ':')) {
        layer.shape.push_back(std::stoull(dim));
    }
    
    // Parse size
    if (!std::getline(iss, token, ',')) return false;
    layer.data_size = std::stoull(token);
    
    // Parse file path
    if (!std::getline(iss, token)) return false;
    layer.file_path = token;
    
    return true;
}
```

};

// Streaming inference engine
class InferenceEngine {
private:
ChunkedModel* model;
std::vector<std::thread> compute_threads;
std::queue<std::function<void()>> task_queue;
std::mutex queue_mutex;
std::condition_variable cv;
bool stop_threads;

```
// Computation cache for intermediate results
struct ComputeCache {
    std::unordered_map<std::string, std::vector<float>> cached_results;
    std::mutex cache_mutex;
    size_t max_cache_entries;
    
    ComputeCache(size_t max_entries = 1000) : max_cache_entries(max_entries) {}
    
    bool get(const std::string& key, std::vector<float>& result) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cached_results.find(key);
        if (it != cached_results.end()) {
            result = it->second;
            return true;
        }
        return false;
    }
    
    void put(const std::string& key, const std::vector<float>& result) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (cached_results.size() >= max_cache_entries) {
            // Simple eviction: remove first entry
            cached_results.erase(cached_results.begin());
        }
        cached_results[key] = result;
    }
};

ComputeCache compute_cache;
```

public:
InferenceEngine(ChunkedModel* m) : model(m), stop_threads(false) {
// Start worker threads
size_t num_threads = std::thread::hardware_concurrency();
for (size_t i = 0; i < num_threads; ++i) {
compute_threads.emplace_back(&InferenceEngine::workerThread, this);
}
}

```
~InferenceEngine() {
    stop_threads = true;
    cv.notify_all();
    
    for (auto& thread : compute_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::vector<float> runInference(const std::vector<float>& input) {
    std::vector<float> current_activations = input;
    
    // Process each layer sequentially
    for (size_t layer_idx = 0; layer_idx < model->getLayerCount(); ++layer_idx) {
        current_activations = processLayer(layer_idx, current_activations);
    }
    
    return current_activations;
}

// Streaming inference for very large inputs
void runStreamingInference(const std::vector<float>& input, 
                          std::function<void(const std::vector<float>&)> callback) {
    
    // Process input in chunks to handle unlimited size
    const size_t input_chunk_size = 1024; // Process 1K elements at a time
    
    for (size_t i = 0; i < input.size(); i += input_chunk_size) {
        size_t chunk_end = std::min(i + input_chunk_size, input.size());
        std::vector<float> chunk(input.begin() + i, input.begin() + chunk_end);
        
        std::vector<float> result = runInference(chunk);
        callback(result);
    }
}
```

private:
void workerThread() {
while (!stop_threads) {
std::unique_lock<std::mutex> lock(queue_mutex);
cv.wait(lock, [this] { return !task_queue.empty() || stop_threads; });

```
        if (stop_threads) break;
        
        auto task = std::move(task_queue.front());
        task_queue.pop();
        lock.unlock();
        
        task();
    }
}

std::vector<float> processLayer(size_t layer_idx, const std::vector<float>& input) {
    // Get layer tensor
    const auto& layer_info = model->getLayerInfo(layer_idx);
    StreamingTensor* tensor = model->getLayer(layer_info.name);
    
    if (!tensor) {
        throw std::runtime_error("Layer tensor not found: " + layer_info.name);
    }
    
    // Check cache first
    std::string cache_key = layer_info.name + "_" + std::to_string(input.size());
    std::vector<float> cached_result;
    if (compute_cache.get(cache_key, cached_result)) {
        return cached_result;
    }
    
    std::vector<float> output;
    output.reserve(input.size()); // Assuming same size for simplicity
    
    // Process tensor chunks
    for (size_t chunk_idx = 0; chunk_idx < tensor->getNumChunks(); ++chunk_idx) {
        void* chunk_data = tensor->getChunk(chunk_idx);
        size_t chunk_size = tensor->getChunkSize(chunk_idx);
        
        if (!chunk_data) {
            throw std::runtime_error("Failed to load tensor chunk");
        }
        
        // Perform computation on chunk (simplified matrix multiplication)
        std::vector<float> chunk_output = computeChunk(
            input, static_cast<float*>(chunk_data), chunk_size / sizeof(float));
        
        output.insert(output.end(), chunk_output.begin(), chunk_output.end());
    }
    
    // Cache result
    compute_cache.put(cache_key, output);
    
    return output;
}

std::vector<float> computeChunk(const std::vector<float>& input, 
                               const float* weights, size_t weight_count) {
    // Simplified computation - replace with actual neural network operations
    std::vector<float> output;
    output.reserve(std::min(input.size(), weight_count));
    
    for (size_t i = 0; i < std::min(input.size(), weight_count); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input.size(); ++j) {
            if (i * input.size() + j < weight_count) {
                sum += input[j] * weights[i * input.size() + j];
            }
        }
        output.push_back(sum);
    }
    
    return output;
}
```

};

// Model converter utility
class ModelConverter {
public:
static bool convertLargeModel(const std::string& input_model_path,
const std::string& output_directory) {
std::cout << “Converting large model to chunked format…” << std::endl;

```
    // Create output directory
    if (system(("mkdir -p " + output_directory).c_str()) != 0) {
        std::cerr << "Failed to create output directory" << std::endl;
        return false;
    }
    
    // This is a simplified converter - you would implement actual model parsing here
    std::ifstream input_file(input_model_path, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input model file" << std::endl;
        return false;
    }
    
    // Get file size
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    
    std::cout << "Input model size: " << (file_size / (1024 * 1024 * 1024)) 
              << " GB" << std::endl;
    
    // Create manifest file
    std::ofstream manifest(output_directory + "/manifest.txt");
    
    // Split into chunks
    size_t chunk_count = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (size_t i = 0; i < chunk_count; ++i) {
        std::string chunk_filename = "layer_" + std::to_string(i) + ".bin";
        std::string chunk_path = output_directory + "/" + chunk_filename;
        
        std::ofstream chunk_file(chunk_path, std::ios::binary);
        if (!chunk_file.is_open()) {
            std::cerr << "Failed to create chunk file: " << chunk_path << std::endl;
            return false;
        }
        
        size_t chunk_size = std::min(CHUNK_SIZE, file_size - i * CHUNK_SIZE);
        std::vector<char> buffer(chunk_size);
        
        input_file.read(buffer.data(), chunk_size);
        chunk_file.write(buffer.data(), chunk_size);
        
        // Write to manifest
        manifest << "layer_" << i << ",1024:1024," << chunk_size 
                 << "," << chunk_filename << std::endl;
        
        std::cout << "Created chunk " << (i + 1) << "/" << chunk_count 
                  << " (" << (chunk_size / (1024 * 1024)) << " MB)" << std::endl;
    }
    
    manifest.close();
    input_file.close();
    
    std::cout << "Model conversion completed successfully!" << std::endl;
    return true;
}
```

};

// Example usage and testing
void example_usage() {
std::cout << “Ollama.cpp Unlimited Memory Demo” << std::endl;
std::cout << “================================” << std::endl;

```
// Convert a large model (if needed)
std::string model_path = "large_model.bin";
std::string chunked_model_dir = "chunked_model";

// Uncomment to convert a model:
// ModelConverter::convertLargeModel(model_path, chunked_model_dir);

// Load chunked model
ChunkedModel model(chunked_model_dir);

// For demo, create a simple manifest if it doesn't exist
std::ofstream demo_manifest(chunked_model_dir + "/manifest.txt");
demo_manifest << "demo_layer,1000:1000,4000000,demo_layer.bin\n";
demo_manifest.close();

// Create demo layer file
std::ofstream demo_layer(chunked_model_dir + "/demo_layer.bin", std::ios::binary);
std::vector<float> demo_data(1000000, 0.5f); // 1M floats
demo_layer.write(reinterpret_cast<const char*>(demo_data.data()), 
                 demo_data.size() * sizeof(float));
demo_layer.close();

if (model.loadModel(chunked_model_dir + "/manifest.txt")) {
    model.printModelInfo();
    
    // Create inference engine
    InferenceEngine engine(&model);
    
    // Run inference with large input
    std::vector<float> large_input(100000, 1.0f); // 100K input
    std::cout << "Running inference on large input..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> result = engine.runInference(large_input);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Output size: " << result.size() << " elements" << std::endl;
    
    // Test streaming inference
    std::cout << "Testing streaming inference..." << std::endl;
    
    size_t callback_count = 0;
    engine.runStreamingInference(large_input, 
        [&callback_count](const std::vector<float>& chunk_result) {
            callback_count++;
            std::cout << "Received streaming result chunk " << callback_count 
                      << " with " << chunk_result.size() << " elements" << std::endl;
        });
}
```

}

int main() {
try {
example_usage();
} catch (const std::exception& e) {
std::cerr << “Error: “ << e.what() << std::endl;
return 1;
}

```
return 0;
```

}