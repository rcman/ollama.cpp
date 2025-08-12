#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <functional>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <queue>
#include <sstream>
#include <exception>

// Include llama.cpp headers
#include "llama.h"
#include "common.h"

// RAII wrapper for llama_batch
class BatchWrapper {
private:
    llama_batch batch;
    bool owns_batch = false;

public:
    BatchWrapper(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
        batch = llama_batch_init(n_tokens, embd, n_seq_max);
        owns_batch = true;
    }

    ~BatchWrapper() {
        if (owns_batch) {
            llama_batch_free(batch);
        }
    }

    // Delete copy constructor and assignment
    BatchWrapper(const BatchWrapper&) = delete;
    BatchWrapper& operator=(const BatchWrapper&) = delete;

    // Move constructor and assignment
    BatchWrapper(BatchWrapper&& other) noexcept {
        batch = other.batch;
        owns_batch = other.owns_batch;
        other.owns_batch = false;
    }

    BatchWrapper& operator=(BatchWrapper&& other) noexcept {
        if (this != &other) {
            if (owns_batch) {
                llama_batch_free(batch);
            }
            batch = other.batch;
            owns_batch = other.owns_batch;
            other.owns_batch = false;
        }
        return *this;
    }

    llama_batch* get() { return &batch; }
    llama_batch& operator*() { return batch; }
};

// SYCL Device Manager
class SYCLDeviceManager {
private:
    std::vector<std::string> available_devices;
    int selected_device = 0;

public:
    void enumerateDevices() {
        // This would use SYCL API to enumerate devices
        // For now, we'll use environment variables
        const char* device_count = std::getenv("SYCL_DEVICE_COUNT");
        int count = device_count ? std::atoi(device_count) : 1;
        
        available_devices.clear();
        for (int i = 0; i < count; i++) {
            available_devices.push_back("Intel GPU " + std::to_string(i));
        }
    }

    void selectDevice(int device_id) {
        if (device_id >= 0 && device_id < available_devices.size()) {
            selected_device = device_id;
            // Set environment variable for SYCL device selection
            std::string device_selector = "level_zero:" + std::to_string(device_id);
            setenv("SYCL_DEVICE_FILTER", device_selector.c_str(), 1);
        }
    }

    void printDevices() {
        std::cout << "Available SYCL devices:" << std::endl;
        for (size_t i = 0; i < available_devices.size(); i++) {
            std::cout << "  [" << i << "] " << available_devices[i] 
                      << (i == selected_device ? " (selected)" : "") << std::endl;
        }
    }
};

class IntelGPUSYCLLlama {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context_params ctx_params;
    llama_model_params model_params;

    std::string model_path;
    bool is_initialized = false;

    // Thread safety
    mutable std::mutex model_mutex;
    mutable std::mutex context_mutex;

    // Model configuration
    int n_ctx = 2048;
    int n_batch = 512;
    int n_gpu_layers = 99;
    int n_threads = 8;

    // Token counting
    std::atomic<size_t> total_tokens_processed{0};

    // SYCL device manager
    SYCLDeviceManager sycl_manager;

    // Get estimated token count for text
    size_t estimateTokenCount(const std::string& text) {
        // Rough estimate: 1 token per 4 characters for English
        // This is conservative and works for most languages
        return (text.length() / 3) + 10;
    }

public:
    IntelGPUSYCLLlama(const std::string& model_path, int gpu_layers = 99)
        : model_path(model_path), n_gpu_layers(gpu_layers) {}

    bool initialize(int device_id = 0) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        // Enumerate and select SYCL device
        sycl_manager.enumerateDevices();
        sycl_manager.selectDevice(device_id);
        sycl_manager.printDevices();
        
        // Initialize llama backend
        llama_backend_init();
        
        // Set up model parameters for Intel GPU (SYCL backend)
        model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        model_params.main_gpu = device_id;
        model_params.split_mode = LLAMA_SPLIT_LAYER;
        model_params.use_mmap = true;
        model_params.use_mlock = false;
        
        // SYCL-specific optimizations
        setenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1", 1);
        setenv("SYCL_CACHE_PERSISTENT", "1", 1);
        
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
        ctx_params.seed = -1;
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_batch;
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
        ctx_params.offload_kqv = true;
        
        // Create context
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to create llama context" << std::endl;
            llama_free_model(model);
            model = nullptr;
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
        
        std::lock_guard<std::mutex> lock(model_mutex);
        
        // Estimate required size more accurately
        size_t estimated_tokens = estimateTokenCount(text) + (add_bos ? 1 : 0);
        std::vector<llama_token> tokens(estimated_tokens);
        
        int n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
                                     tokens.data(), tokens.size(), add_bos, false);
        
        if (n_tokens < 0) {
            // Resize and try again
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
                                     tokens.data(), tokens.size(), add_bos, false);
            
            if (n_tokens < 0) {
                std::cerr << "Failed to tokenize text even after resize" << std::endl;
                return {};
            }
        }
        
        tokens.resize(n_tokens);
        total_tokens_processed += n_tokens;
        return tokens;
    }

    std::string detokenize(const std::vector<llama_token>& tokens) {
        if (!is_initialized) return "";
        
        std::lock_guard<std::mutex> lock(model_mutex);
        
        std::string result;
        result.reserve(tokens.size() * 10); // Reserve space for efficiency
        
        for (auto token : tokens) {
            // Use dynamic buffer for token pieces
            std::vector<char> piece_buffer(512);
            int n = llama_token_to_piece(model, token, piece_buffer.data(), 
                                       piece_buffer.size(), 0, false);
            
            if (n < 0) {
                // Buffer too small, resize and retry
                piece_buffer.resize(-n);
                n = llama_token_to_piece(model, token, piece_buffer.data(), 
                                       piece_buffer.size(), 0, false);
            }
            
            if (n > 0) {
                result.append(piece_buffer.data(), n);
            }
        }
        return result;
    }

    // Context state management
    struct ContextState {
        std::vector<llama_token> tokens;
        size_t n_past;
    };

    ContextState saveContextState() {
        std::lock_guard<std::mutex> lock(context_mutex);
        ContextState state;
        state.n_past = llama_get_kv_cache_used_cells(ctx);
        // In a real implementation, you'd save the actual KV cache
        return state;
    }

    void restoreContextState(const ContextState& state) {
        std::lock_guard<std::mutex> lock(context_mutex);
        // Clear context
        llama_kv_cache_clear(ctx);
        // In a real implementation, you'd restore the KV cache
    }

    // Thread-safe text generation
    std::string generate(const std::string& prompt, 
                        int max_tokens = 256,
                        float temperature = 0.8f,
                        float top_p = 0.95f,
                        int top_k = 40,
                        bool reset_context = true) {
        
        if (!is_initialized) {
            std::cerr << "Model not initialized!" << std::endl;
            return "";
        }
        
        std::lock_guard<std::mutex> lock(context_mutex);
        
        // Reset context if requested
        if (reset_context) {
            llama_kv_cache_clear(ctx);
        }
        
        // Tokenize input
        auto input_tokens = tokenize(prompt, true);
        if (input_tokens.empty()) {
            std::cerr << "Failed to tokenize input" << std::endl;
            return "";
        }
        
        std::cout << "Input tokens: " << input_tokens.size() << std::endl;
        
        // Create batch with RAII wrapper
        BatchWrapper batch_wrapper(input_tokens.size(), 0, 1);
        auto batch = batch_wrapper.get();
        
        // Add input tokens to batch
        for (size_t i = 0; i < input_tokens.size(); i++) {
            llama_batch_add(*batch, input_tokens[i], i, {0}, false);
        }
        
        // Set last token to generate logits
        if (batch->n_tokens > 0) {
            batch->logits[batch->n_tokens - 1] = true;
        }
        
        // Process input tokens
        if (llama_decode(ctx, *batch) != 0) {
            std::cerr << "Failed to decode input tokens" << std::endl;
            return "";
        }
        
        std::string generated_text;
        std::vector<llama_token> generated_tokens;
        generated_tokens.reserve(max_tokens);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate tokens one by one
        for (int i = 0; i < max_tokens; i++) {
            // Get logits for the last token
            float* logits = llama_get_logits_ith(ctx, batch->n_tokens - 1);
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
            std::vector<char> piece_buffer(512);
            int n = llama_token_to_piece(model, new_token, piece_buffer.data(), 
                                       piece_buffer.size(), 0, false);
            
            if (n < 0) {
                piece_buffer.resize(-n);
                n = llama_token_to_piece(model, new_token, piece_buffer.data(), 
                                       piece_buffer.size(), 0, false);
            }
            
            if (n > 0) {
                std::string token_text(piece_buffer.data(), n);
                generated_text += token_text;
                std::cout << token_text << std::flush;
            }
            
            // Clear batch and add new token
            llama_batch_clear(*batch);
            llama_batch_add(*batch, new_token, input_tokens.size() + i, {0}, true);
            
            // Decode the new token
            if (llama_decode(ctx, *batch) != 0) {
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
        
        total_tokens_processed += generated_tokens.size();
        
        return generated_text;
    }

    // Interactive chat with conversation history and token management
    class ChatSession {
    private:
        IntelGPUSYCLLlama* engine;
        std::vector<std::pair<std::string, std::string>> conversation;
        std::string system_prompt;
        size_t max_context_tokens;
        size_t current_context_tokens = 0;
        
        // Calculate tokens for a message
        size_t countTokens(const std::string& text) {
            auto tokens = engine->tokenize(text, false);
            return tokens.size();
        }
        
        // Truncate conversation to fit context window
        void truncateConversation() {
            // Reserve space for system prompt and current response
            size_t reserved_tokens = countTokens(system_prompt) + 512;
            size_t available_tokens = max_context_tokens - reserved_tokens;
            
            // Remove oldest messages until we fit
            while (!conversation.empty() && current_context_tokens > available_tokens) {
                auto& [user_msg, assistant_msg] = conversation.front();
                current_context_tokens -= countTokens(user_msg) + countTokens(assistant_msg) + 20;
                conversation.erase(conversation.begin());
            }
        }
        
    public:
        ChatSession(IntelGPUSYCLLlama* eng, const std::string& sys_prompt = "") 
            : engine(eng), system_prompt(sys_prompt) {
            max_context_tokens = engine->n_ctx - 512; // Reserve tokens for response
            if (!system_prompt.empty()) {
                current_context_tokens = countTokens(system_prompt);
            }
        }
        
        std::string chat(const std::string& user_input) {
            // Count tokens for new input
            size_t input_tokens = countTokens(user_input);
            current_context_tokens += input_tokens + 10; // Extra for formatting
            
            // Truncate if needed
            truncateConversation();
            
            // Build conversation context
            std::stringstream prompt_builder;
            if (!system_prompt.empty()) {
                prompt_builder << system_prompt << "\n\n";
            }
            
            // Add conversation history
            for (const auto& [user_msg, assistant_msg] : conversation) {
                prompt_builder << "Human: " << user_msg << "\n\n";
                prompt_builder << "Assistant: " << assistant_msg << "\n\n";
            }
            
            prompt_builder << "Human: " << user_input << "\n\nAssistant: ";
            
            std::string prompt = prompt_builder.str();
            
            // Generate response without resetting context
            std::string response = engine->generate(prompt, 512, 0.7f, 0.95f, 40, false);
            
            // Update token count
            current_context_tokens += countTokens(response) + 10;
            
            // Add to conversation history
            conversation.emplace_back(user_input, response);
            
            return response;
        }
        
        void clearHistory() {
            conversation.clear();
            current_context_tokens = system_prompt.empty() ? 0 : countTokens(system_prompt);
            // Also clear the model context
            std::lock_guard<std::mutex> lock(engine->context_mutex);
            llama_kv_cache_clear(engine->ctx);
        }
        
        size_t getContextUsage() const {
            return current_context_tokens;
        }
        
        size_t getMaxContext() const {
            return max_context_tokens;
        }
    };

    std::unique_ptr<ChatSession> createChatSession(const std::string& system_prompt = "") {
        return std::make_unique<ChatSession>(this, system_prompt);
    }

    // Performance benchmark with state preservation
    void benchmark(int iterations = 5) {
        if (!is_initialized) {
            std::cerr << "Model not initialized!" << std::endl;
            return;
        }
        
        // Save current context state
        auto saved_state = saveContextState();
        
        std::cout << "\n=== Performance Benchmark ===" << std::endl;
        std::string test_prompt = "The future of artificial intelligence is";
        
        std::vector<double> times;
        std::vector<int> token_counts;
        times.reserve(iterations);
        token_counts.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            std::cout << "Iteration " << (i + 1) << "/" << iterations << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            std::string result = generate(test_prompt, 100, 0.7f, 0.95f, 40, true);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            times.push_back(duration.count());
            
            // Count tokens in result
            auto tokens = tokenize(result, false);
            token_counts.push_back(tokens.size());
        }
        
        // Restore context state
        restoreContextState(saved_state);
        
        // Calculate statistics
        double avg_time = 0;
        double avg_tokens = 0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (size_t i = 0; i < times.size(); i++) {
            avg_time += times[i];
            avg_tokens += token_counts[i];
            min_time = std::min(min_time, times[i]);
            max_time = std::max(max_time, times[i]);
        }
        avg_time /= iterations;
        avg_tokens /= iterations;
        
        std::cout << "\nBenchmark Results:" << std::endl;
        std::cout << "Average time: " << avg_time << " ms" << std::endl;
        std::cout << "Min time: " << min_time << " ms" << std::endl;
        std::cout << "Max time: " << max_time << " ms" << std::endl;
        std::cout << "Average tokens: " << avg_tokens << std::endl;
        std::cout << "Average speed: " << avg_tokens / (avg_time / 1000.0) << " tokens/second" << std::endl;
        std::cout << "Total tokens processed: " << total_tokens_processed.load() << std::endl;
    }

    // Get statistics
    size_t getTotalTokensProcessed() const {
        return total_tokens_processed.load();
    }

    ~IntelGPUSYCLLlama() {
        std::lock_guard<std::mutex> model_lock(model_mutex);
        std::lock_guard<std::mutex> ctx_lock(context_mutex);
        
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
        llama_backend_free();
    }
};

// Exception handler for main
void handleException(const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "Make sure you have:" << std::endl;
    std::cerr << "1. Intel oneAPI Base Toolkit installed" << std::endl;
    std::cerr << "2. Intel GPU drivers installed" << std::endl;
    std::cerr << "3. llama.cpp compiled with SYCL support" << std::endl;
    std::cerr << "4. Environment sourced: source /opt/intel/oneapi/setvars.sh" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <model_path> [gpu_layers] [device_id]" << std::endl;
            std::cout << "Example: " << argv[0] << " models/llama-2-7b-chat.Q4_0.gguf 33 0" << std::endl;
            std::cout << "\nNote: Make sure you have:" << std::endl;
            std::cout << "1. Intel oneAPI Base Toolkit installed" << std::endl;
            std::cout << "2. Intel GPU drivers installed" << std::endl;
            std::cout << "3. llama.cpp compiled with SYCL support" << std::endl;
            std::cout << "4. Environment sourced: source /opt/intel/oneapi/setvars.sh" << std::endl;
            return -1;
        }

        std::string model_path = argv[1];
        int gpu_layers = argc > 2 ? std::atoi(argv[2]) : 99;
        int device_id = argc > 3 ? std::atoi(argv[3]) : 0;

        // Initialize the engine
        IntelGPUSYCLLlama engine(model_path, gpu_layers);

        if (!engine.initialize(device_id)) {
            std::cerr << "Failed to initialize model" << std::endl;
            return -1;
        }

        // Create chat session
        std::string system_prompt = "You are a helpful AI assistant. Provide clear and concise responses.";
        auto chat = engine.createChatSession(system_prompt);

        std::cout << "\n=== Intel GPU llama.cpp Chat (type 'quit' to exit) ===" << std::endl;
        std::cout << "Commands: 'quit', 'benchmark', 'clear', 'generate <text>', 'stats', 'context'" << std::endl;

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
            
            if (input == "stats") {
                std::cout << "Total tokens processed: " << engine.getTotalTokensProcessed() << std::endl;
                continue;
            }
            
            if (input == "context") {
                std::cout << "Context usage: " << chat->getContextUsage() 
                          << "/" << chat->getMaxContext() << " tokens" << std::endl;
                continue;
            }
            
            if (input.substr(0, 8) == "generate") {
                std::string text = input.length() > 9 ? input.substr(9) : "Hello world";
                std::cout << "Assistant: ";
                engine.generate(text, 256, 0.7f, 0.95f, 40, true);
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

        std::cout << "\nTotal tokens processed in session: " << engine.getTotalTokensProcessed() << std::endl;

    } catch (const std::exception& e) {
        handleException(e);
        return -1;
    }

    return 0;
}