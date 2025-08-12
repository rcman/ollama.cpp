#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <functional>

// Include llama.cpp headers - these are the actual current headers
#include “llama.h”
#include “common.h”

class IntelGPUSYCLLlama {
private:
llama_model* model = nullptr;
llama_context* ctx = nullptr;
llama_context_params ctx_params;
llama_model_params model_params;

```
std::string model_path;
bool is_initialized = false;

// Model configuration
int n_ctx = 2048;
int n_batch = 512;
int n_gpu_layers = 99;  // Use 99 to offload all layers to GPU
int n_threads = 8;
```

public:
IntelGPUSYCLLlama(const std::string& model_path, int gpu_layers = 99)
: model_path(model_path), n_gpu_layers(gpu_layers) {}

```
bool initialize() {
    // Initialize llama backend
    llama_backend_init();
    
    // Set up model parameters for Intel GPU (SYCL backend)
    model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;  // Offload layers to Intel GPU
    model_params.main_gpu = 0;                 // Use first Intel GPU device
    model_params.split_mode = LLAMA_SPLIT_LAYER;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    
    std::cout << "Loading model: " << model_path << std::endl;
    std::cout << "GPU layers to offload: " << n_gpu_layers << std::endl;
    
    // Load model
    model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return false;
    }
    
    // Set up context parameters
    ctx_params = llama_context_default_params();
    ctx_params.seed = -1;                    // Random seed
    ctx_params.n_ctx = n_ctx;               // Context size
    ctx_params.n_batch = n_batch;           // Batch size for prompt processing
    ctx_params.n_threads = n_threads;       // CPU threads
    ctx_params.n_threads_batch = n_threads; // Threads for batch processing
    ctx_params.offload_kqv = true;          // Offload KV cache to GPU
    
    // Create context
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create llama context" << std::endl;
        return false;
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Context size: " << llama_n_ctx(ctx) << std::endl;
    std::cout << "Vocab size: " << llama_n_vocab(model) << std::endl;
    
    is_initialized = true;
    return true;
}

std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true) {
    if (!is_initialized) return {};
    
    // Calculate required size
    int n_tokens = text.length() + (add_bos ? 1 : 0);
    std::vector<llama_token> tokens(n_tokens);
    
    n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
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
    if (!is_initialized) return "";
    
    std::string result;
    for (auto token : tokens) {
        char piece[256];
        int n = llama_token_to_piece(model, token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            result += std::string(piece, n);
        }
    }
    return result;
}

// Simple text generation
std::string generate(const std::string& prompt, 
                    int max_tokens = 256,
                    float temperature = 0.8f,
                    float top_p = 0.95f,
                    int top_k = 40) {
    
    if (!is_initialized) {
        std::cerr << "Model not initialized!" << std::endl;
        return "";
    }
    
    // Tokenize input
    auto input_tokens = tokenize(prompt, true);
    std::cout << "Input tokens: " << input_tokens.size() << std::endl;
    
    // Create batch for input tokens
    llama_batch batch = llama_batch_init(std::max(input_tokens.size(), (size_t)n_batch), 0, 1);
    
    // Add input tokens to batch
    for (size_t i = 0; i < input_tokens.size(); i++) {
        llama_batch_add(batch, input_tokens[i], i, {0}, false);
    }
    
    // Set last token to generate logits
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }
    
    // Process input tokens
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to decode input tokens" << std::endl;
        llama_batch_free(batch);
        return "";
    }
    
    std::string generated_text;
    std::vector<llama_token> generated_tokens;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Generate tokens one by one
    for (int i = 0; i < max_tokens; i++) {
        // Get logits for the last token
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        int n_vocab = llama_n_vocab(model);
        
        // Create candidates array
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        
        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
        
        // Apply sampling
        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
        llama_sample_temp(ctx, &candidates_p, temperature);
        
        // Sample the next token
        llama_token new_token = llama_sample_token(ctx, &candidates_p);
        
        // Check for end of sequence
        if (llama_token_is_eog(model, new_token)) {
            break;
        }
        
        generated_tokens.push_back(new_token);
        
        // Convert token to text and add to result
        char piece[256];
        int n = llama_token_to_piece(model, new_token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            std::string token_text(piece, n);
            generated_text += token_text;
            std::cout << token_text << std::flush; // Stream output
        }
        
        // Clear batch and add new token
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, input_tokens.size() + i, {0}, true);
        
        // Decode the new token
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode token" << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::endl;
    std::cout << "Generated " << generated_tokens.size() << " tokens in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Speed: " << (float)generated_tokens.size() / (duration.count() / 1000.0f) 
              << " tokens/second" << std::endl;
    
    llama_batch_free(batch);
    return generated_text;
}

// Interactive chat with conversation history
class ChatSession {
private:
    IntelGPUSYCLLlama* engine;
    std::vector<std::pair<std::string, std::string>> conversation;
    std::string system_prompt;
    
public:
    ChatSession(IntelGPUSYCLLlama* eng, const std::string& sys_prompt = "") 
        : engine(eng), system_prompt(sys_prompt) {}
    
    std::string chat(const std::string& user_input) {
        // Build conversation context
        std::string prompt = system_prompt;
        
        // Add conversation history
        for (const auto& [user_msg, assistant_msg] : conversation) {
            prompt += "\n\nHuman: " + user_msg;
            prompt += "\n\nAssistant: " + assistant_msg;
        }
        
        prompt += "\n\nHuman: " + user_input + "\n\nAssistant: ";
        
        // Generate response
        std::string response = engine->generate(prompt, 512, 0.7f, 0.95f, 40);
        
        // Add to conversation history
        conversation.emplace_back(user_input, response);
        
        // Keep only last 5 exchanges to manage context length
        if (conversation.size() > 5) {
            conversation.erase(conversation.begin());
        }
        
        return response;
    }
    
    void clearHistory() {
        conversation.clear();
    }
};

std::unique_ptr<ChatSession> createChatSession(const std::string& system_prompt = "") {
    return std::make_unique<ChatSession>(this, system_prompt);
}

// Performance benchmark
void benchmark(int iterations = 5) {
    if (!is_initialized) {
        std::cerr << "Model not initialized!" << std::endl;
        return;
    }
    
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    std::string test_prompt = "The future of artificial intelligence is";
    
    std::vector<double> times;
    std::vector<int> token_counts;
    
    for (int i = 0; i < iterations; i++) {
        std::cout << "Iteration " << (i + 1) << "/" << iterations << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = generate(test_prompt, 100, 0.7f);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        times.push_back(duration.count());
        
        // Count tokens in result
        auto tokens = tokenize(result, false);
        token_counts.push_back(tokens.size());
    }
    
    // Calculate averages
    double avg_time = 0;
    double avg_tokens = 0;
    for (size_t i = 0; i < times.size(); i++) {
        avg_time += times[i];
        avg_tokens += token_counts[i];
    }
    avg_time /= iterations;
    avg_tokens /= iterations;
    
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Average tokens: " << avg_tokens << std::endl;
    std::cout << "Average speed: " << avg_tokens / (avg_time / 1000.0) << " tokens/second" << std::endl;
}

~IntelGPUSYCLLlama() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_free_model(model);
    }
    llama_backend_free();
}
```

};

int main(int argc, char* argv[]) {
if (argc < 2) {
std::cout << “Usage: “ << argv[0] << “ <model_path> [gpu_layers]” << std::endl;
std::cout << “Example: “ << argv[0] << “ models/llama-2-7b-chat.Q4_0.gguf 33” << std::endl;
std::cout << “\nNote: Make sure you have:” << std::endl;
std::cout << “1. Intel oneAPI Base Toolkit installed” << std::endl;
std::cout << “2. Intel GPU drivers installed” << std::endl;
std::cout << “3. llama.cpp compiled with SYCL support” << std::endl;
std::cout << “4. Environment sourced: source /opt/intel/oneapi/setvars.sh” << std::endl;
return -1;
}

```
std::string model_path = argv[1];
int gpu_layers = argc > 2 ? std::atoi(argv[2]) : 99;  // Default to all layers

// Initialize the engine
IntelGPUSYCLLlama engine(model_path, gpu_layers);

if (!engine.initialize()) {
    std::cerr << "Failed to initialize model" << std::endl;
    return -1;
}

// Create chat session
std::string system_prompt = "You are a helpful AI assistant. Provide clear and concise responses.";
auto chat = engine.createChatSession(system_prompt);

std::cout << "\n=== Intel GPU llama.cpp Chat (type 'quit' to exit) ===" << std::endl;
std::cout << "Commands: 'quit', 'benchmark', 'clear', 'generate <text>'" << std::endl;

std::string input;
while (true) {
    std::cout << "\nUser: ";
    std::getline(std::cin, input);
    
    if (input == "quit" || input == "exit") {
        break;
    }
    
    if (input == "benchmark") {
        engine.benchmark();
        continue;
    }
    
    if (input == "clear") {
        chat->clearHistory();
        std::cout << "Chat history cleared." << std::endl;
        continue;
    }
    
    if (input.substr(0, 8) == "generate") {
        std::string text = input.length() > 9 ? input.substr(9) : "Hello world";
        std::cout << "Assistant: ";
        engine.generate(text, 256);
        std::cout << std::endl;
        continue;
    }
    
    if (input.empty()) {
        continue;
    }
    
    std::cout << "Assistant: ";
    chat->chat(input);
    std::cout << std::endl;
}

return 0;
```

}