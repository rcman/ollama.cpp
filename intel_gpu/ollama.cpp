#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <CL/cl.h>

// Include llama.cpp headers
#include “llama.h”
#include “common.h”
#include “ggml.h”
#include “ggml-opencl.h”

class IntelGPULlamaEngine {
private:
llama_model* model = nullptr;
llama_context* ctx = nullptr;
llama_context_params ctx_params;
llama_model_params model_params;

```
// OpenCL context for Intel GPU
cl_platform_id platform;
cl_device_id device;
cl_context cl_context;
cl_command_queue queue;

// Model configuration
std::string model_path;
int n_ctx = 2048;
int n_batch = 512;
int n_gpu_layers = 35;  // Number of layers to offload to GPU
int n_threads = 8;

// Chat state
std::vector<llama_token> tokens_list;
std::string system_prompt;
bool is_initialized = false;
```

public:
IntelGPULlamaEngine(const std::string& model_path, int gpu_layers = 35)
: model_path(model_path), n_gpu_layers(gpu_layers) {
initializeOpenCL();
}

```
bool initializeOpenCL() {
    // Get Intel OpenCL platform
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
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
        std::cerr << "No Intel GPU found\n";
        return false;
    }
    
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
    // Print device info
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Using Intel GPU: " << device_name << std::endl;
    
    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
    std::cout << "GPU Memory: " << mem_size / (1024*1024*1024) << " GB" << std::endl;
    
    // Create OpenCL context
    cl_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(cl_context, device, 0, nullptr);
    
    return true;
}

bool initializeModel() {
    // Initialize llama backend
    llama_backend_init(false);
    
    // Set up model parameters for Intel GPU acceleration
    model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.main_gpu = 0;
    model_params.split_mode = LLAMA_SPLIT_LAYER;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    
    // Load model
    std::cout << "Loading model: " << model_path << std::endl;
    std::cout << "GPU layers: " << n_gpu_layers << std::endl;
    
    model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return false;
    }
    
    // Set up context parameters
    ctx_params = llama_context_default_params();
    ctx_params.seed = -1;  // Random seed
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.logits_all = false;
    ctx_params.embedding = false;
    ctx_params.offload_kqv = true;  // Offload KV cache to GPU
    
    // Create context
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create llama context" << std::endl;
        return false;
    }
    
    // Initialize tokenizer
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Context size: " << llama_n_ctx(ctx) << std::endl;
    std::cout << "Model size: " << llama_model_size(model) / (1024*1024) << " MB" << std::endl;
    
    is_initialized = true;
    return true;
}

void setSystemPrompt(const std::string& prompt) {
    system_prompt = prompt;
}

std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true) {
    std::vector<llama_token> tokens(text.length() + (add_bos ? 1 : 0));
    int n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
                                 tokens.data(), tokens.size(), add_bos, false);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
                                 tokens.data(), tokens.size(), add_bos, false);
    }
    
    tokens.resize(n_tokens);
    return tokens;
}

std::string detokenize(const std::vector<llama_token>& tokens) {
    std::string result;
    for (auto token : tokens) {
        char piece[256];
        int n = llama_token_to_piece(model, token, piece, sizeof(piece));
        if (n > 0) {
            result += std::string(piece, n);
        }
    }
    return result;
}

// Streaming generation with callback
void generateStreaming(const std::string& prompt, 
                      std::function<bool(const std::string&)> callback,
                      int max_tokens = 512,
                      float temperature = 0.7f,
                      float top_p = 0.95f,
                      int top_k = 40) {
    
    if (!is_initialized) {
        std::cerr << "Model not initialized!" << std::endl;
        return;
    }
    
    // Prepare full prompt with system message
    std::string full_prompt = system_prompt.empty() ? prompt : system_prompt + "\n\n" + prompt;
    
    // Tokenize input
    auto input_tokens = tokenize(full_prompt, true);
    
    // Evaluate input tokens
    if (llama_decode(ctx, llama_batch_get_one(input_tokens.data(), input_tokens.size(), 0, 0))) {
        std::cerr << "Failed to decode input tokens" << std::endl;
        return;
    }
    
    // Generate tokens
    std::vector<llama_token> generated_tokens;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < max_tokens; i++) {
        // Get logits
        float* logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_n_vocab(model);
        
        // Apply temperature and top-k/top-p sampling
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        
        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
        
        // Apply top-k sampling
        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
        
        // Apply top-p sampling
        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
        
        // Apply temperature
        llama_sample_temp(ctx, &candidates_p, temperature);
        
        // Sample token
        llama_token new_token = llama_sample_token(ctx, &candidates_p);
        
        // Check for end of sequence
        if (new_token == llama_token_eos(model)) {
            break;
        }
        
        generated_tokens.push_back(new_token);
        
        // Convert token to text and call callback
        std::string token_text = detokenize({new_token});
        if (!callback(token_text)) {
            break; // Callback returned false, stop generation
        }
        
        // Prepare for next iteration
        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1, input_tokens.size() + i, 0))) {
            std::cerr << "Failed to decode token" << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    float tokens_per_second = (float)generated_tokens.size() / (duration.count() / 1000.0f);
    std::cout << "\nGeneration stats:" << std::endl;
    std::cout << "Tokens generated: " << generated_tokens.size() << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Tokens/second: " << tokens_per_second << std::endl;
}

// Batch processing for multiple prompts
std::vector<std::string> generateBatch(const std::vector<std::string>& prompts,
                                      int max_tokens = 512,
                                      float temperature = 0.7f) {
    std::vector<std::string> results;
    
    for (const auto& prompt : prompts) {
        std::string result;
        generateStreaming(prompt, [&result](const std::string& token) {
            result += token;
            return true; // Continue generation
        }, max_tokens, temperature);
        
        results.push_back(result);
    }
    
    return results;
}

// Chat interface with conversation history
class ChatSession {
private:
    IntelGPULlamaEngine* engine;
    std::vector<std::pair<std::string, std::string>> conversation_history;
    std::string system_message;
    
public:
    ChatSession(IntelGPULlamaEngine* eng, const std::string& system_msg = "") 
        : engine(eng), system_message(system_msg) {}
    
    void addSystemMessage(const std::string& message) {
        system_message = message;
    }
    
    std::string chat(const std::string& user_input, 
                    std::function<bool(const std::string&)> callback = nullptr) {
        // Build conversation context
        std::string context = system_message.empty() ? "" : system_message + "\n\n";
        
        for (const auto& [user_msg, assistant_msg] : conversation_history) {
            context += "Human: " + user_msg + "\n";
            context += "Assistant: " + assistant_msg + "\n";
        }
        
        context += "Human: " + user_input + "\nAssistant: ";
        
        std::string response;
        engine->generateStreaming(context, [&response, &callback](const std::string& token) {
            response += token;
            if (callback) {
                return callback(token);
            }
            return true;
        });
        
        // Add to conversation history
        conversation_history.emplace_back(user_input, response);
        
        // Trim history if too long (keep last 10 exchanges)
        if (conversation_history.size() > 10) {
            conversation_history.erase(conversation_history.begin());
        }
        
        return response;
    }
    
    void clearHistory() {
        conversation_history.clear();
    }
    
    void printHistory() {
        for (const auto& [user_msg, assistant_msg] : conversation_history) {
            std::cout << "Human: " << user_msg << std::endl;
            std::cout << "Assistant: " << assistant_msg << std::endl;
            std::cout << "---" << std::endl;
        }
    }
};

std::unique_ptr<ChatSession> createChatSession(const std::string& system_message = "") {
    return std::make_unique<ChatSession>(this, system_message);
}

// Model information and statistics
void printModelInfo() {
    if (!is_initialized) return;
    
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Context size: " << llama_n_ctx(ctx) << std::endl;
    std::cout << "Vocab size: " << llama_n_vocab(model) << std::endl;
    std::cout << "GPU layers: " << n_gpu_layers << std::endl;
    std::cout << "Batch size: " << n_batch << std::endl;
    std::cout << "Threads: " << n_threads << std::endl;
    
    // GPU memory usage estimation
    size_t model_size = llama_model_size(model);
    std::cout << "Model size: " << model_size / (1024*1024) << " MB" << std::endl;
    
    // Print GPU device info again
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "GPU Device: " << device_name << std::endl;
}

// Performance benchmark
void benchmark(int num_iterations = 10) {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    std::string test_prompt = "Explain quantum computing in simple terms.";
    std::vector<double> generation_times;
    std::vector<int> token_counts;
    
    for (int i = 0; i < num_iterations; i++) {
        std::cout << "Iteration " << (i + 1) << "/" << num_iterations << std::endl;
        
        int tokens_generated = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        generateStreaming(test_prompt, [&tokens_generated](const std::string& token) {
            tokens_generated++;
            return tokens_generated < 100; // Generate 100 tokens
        }, 100);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        generation_times.push_back(duration.count());
        token_counts.push_back(tokens_generated);
    }
    
    // Calculate averages
    double avg_time = 0;
    double avg_tokens = 0;
    for (int i = 0; i < num_iterations; i++) {
        avg_time += generation_times[i];
        avg_tokens += token_counts[i];
    }
    avg_time /= num_iterations;
    avg_tokens /= num_iterations;
    
    double avg_tokens_per_second = avg_tokens / (avg_time / 1000.0);
    
    std::cout << "Average generation time: " << avg_time << " ms" << std::endl;
    std::cout << "Average tokens generated: " << avg_tokens << std::endl;
    std::cout << "Average tokens/second: " << avg_tokens_per_second << std::endl;
}

~IntelGPULlamaEngine() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_free_model(model);
    }
    llama_backend_free();
    
    if (queue) clReleaseCommandQueue(queue);
    if (cl_context) clReleaseContext(cl_context);
}
```

};

// Example usage and main function
int main(int argc, char* argv[]) {
if (argc < 2) {
std::cout << “Usage: “ << argv[0] << “ <model_path> [gpu_layers]” << std::endl;
std::cout << “Example: “ << argv[0] << “ models/llama-2-7b-chat.q4_0.gguf 35” << std::endl;
return -1;
}

```
std::string model_path = argv[1];
int gpu_layers = argc > 2 ? std::atoi(argv[2]) : 35;

// Initialize the engine
IntelGPULlamaEngine engine(model_path, gpu_layers);

if (!engine.initializeModel()) {
    std::cerr << "Failed to initialize model" << std::endl;
    return -1;
}

engine.printModelInfo();

// Set up system prompt
std::string system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and concise responses.";
engine.setSystemPrompt(system_prompt);

// Create chat session
auto chat = engine.createChatSession(system_prompt);

// Interactive chat loop
std::cout << "\n=== Interactive Chat (type 'quit' to exit, 'benchmark' to run performance test) ===" << std::endl;
std::string user_input;

while (true) {
    std::cout << "\nHuman: ";
    std::getline(std::cin, user_input);
    
    if (user_input == "quit" || user_input == "exit") {
        break;
    }
    
    if (user_input == "benchmark") {
        engine.benchmark();
        continue;
    }
    
    if (user_input == "clear") {
        chat->clearHistory();
        std::cout << "Chat history cleared." << std::endl;
        continue;
    }
    
    if (user_input == "history") {
        chat->printHistory();
        continue;
    }
    
    if (user_input.empty()) {
        continue;
    }
    
    std::cout << "Assistant: ";
    
    // Stream the response with real-time output
    chat->chat(user_input, [](const std::string& token) {
        std::cout << token << std::flush;
        return true; // Continue generation
    });
    
    std::cout << std::endl;
}

return 0;
```

}