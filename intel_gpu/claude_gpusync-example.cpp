// Advanced Intel GPU llama.cpp Integration Examples
// This file demonstrates advanced features and integration patterns

#include “intel_gpu_ollama.hpp”
#include <json/json.h>  // For JSON handling
#include <websocketpp/config/asio_no_tls.hpp>  // For WebSocket server
#include <websocketpp/server.hpp>
#include <fstream>
#include <regex>

// WebSocket server for real-time chat API
class LlamaWebSocketServer {
private:
websocketpp::server<websocketpp::config::asio> server;
std::unique_ptr<IntelGPULlamaEngine> engine;
std::map<websocketpp::connection_hdl, std::unique_ptr<IntelGPULlamaEngine::ChatSession>,
std::owner_less<websocketpp::connection_hdl>> sessions;

public:
LlamaWebSocketServer(const std::string& model_path, int port = 8080) {
engine = std::make_unique<IntelGPULlamaEngine>(model_path, 35);
if (!engine->initializeModel()) {
throw std::runtime_error(“Failed to initialize model”);
}

```
    server.set_access_channels(websocketpp::log::alevel::all);
    server.clear_access_channels(websocketpp::log::alevel::frame_payload);
    server.init
```