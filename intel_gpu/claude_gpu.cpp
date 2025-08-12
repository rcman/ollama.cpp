#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include <iostream>

class SYCLDeviceManager {
private:
    std::vector<sycl::device> sycl_devices;
    std::vector<std::string> available_devices;
    int selected_device = 0;

public:
    void enumerateDevices() {
        available_devices.clear();
        sycl_devices.clear();
        
        // Get all available SYCL devices
        auto all_devices = sycl::device::get_devices();
        
        for (size_t i = 0; i < all_devices.size(); i++) {
            auto& device = all_devices[i];
            
            // Get device info
            std::string device_name = device.get_info<sycl::info::device::name>();
            std::string vendor = device.get_info<sycl::info::device::vendor>();
            auto device_type = device.get_info<sycl::info::device::device_type>();
            
            // Build device description
            std::string type_str;
            switch (device_type) {
                case sycl::info::device_type::gpu:
                    type_str = "GPU";
                    break;
                case sycl::info::device_type::cpu:
                    type_str = "CPU";
                    break;
                case sycl::info::device_type::accelerator:
                    type_str = "Accelerator";
                    break;
                default:
                    type_str = "Unknown";
            }
            
            std::string device_desc = vendor + " " + device_name + " (" + type_str + ")";
            available_devices.push_back(device_desc);
            
            // Store the actual SYCL device
            sycl_devices.push_back(device);
        }
        
        // If no devices found, that's a problem
        if (sycl_devices.empty()) {
            std::cerr << "Warning: No SYCL devices found!" << std::endl;
            std::cerr << "Make sure Intel GPU drivers and Level Zero are properly installed." << std::endl;
        }
    }

    void selectDevice(int device_id) {
        if (device_id >= 0 && device_id < sycl_devices.size()) {
            selected_device = device_id;
            
            // For llama.cpp SYCL backend, we need to set the device selector
            // The ONEAPI_DEVICE_SELECTOR format should be:
            // "level_zero:gpu" for Intel GPUs, or specific device index
            std::string device_selector;
            
            auto device_type = sycl_devices[device_id].get_info<sycl::info::device::device_type>();
            if (device_type == sycl::info::device_type::gpu) {
                // For Intel Arc GPUs, use level_zero backend
                device_selector = "level_zero:" + std::to_string(device_id);
            } else {
                // For other devices (CPU, etc)
                device_selector = "opencl:" + std::to_string(device_id);
            }
            
            setenv("ONEAPI_DEVICE_SELECTOR", device_selector.c_str(), 1);
            
            // Also set SYCL-specific environment variables for performance
            setenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1", 1);
            setenv("SYCL_CACHE_PERSISTENT", "1", 1);
            
            // For Intel Arc GPUs, additional optimizations
            if (device_type == sycl::info::device_type::gpu) {
                setenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE", 1);
            }
        }
    }

    void printDevices() {
        std::cout << "\nAvailable SYCL devices:" << std::endl;
        std::cout << "======================" << std::endl;
        
        if (available_devices.empty()) {
            std::cout << "No SYCL devices found!" << std::endl;
            return;
        }
        
        for (size_t i = 0; i < available_devices.size(); i++) {
            std::cout << "  [" << i << "] " << available_devices[i];
            
            if (i < sycl_devices.size()) {
                // Print additional device info
                auto& device = sycl_devices[i];
                
                // Memory info
                auto global_mem = device.get_info<sycl::info::device::global_mem_size>();
                std::cout << "\n      Memory: " << (global_mem / (1024*1024*1024)) << " GB";
                
                // Compute units
                auto compute_units = device.get_info<sycl::info::device::max_compute_units>();
                std::cout << ", Compute Units: " << compute_units;
                
                // Max work group size
                auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
                std::cout << ", Max Work Group: " << max_wg_size;
            }
            
            if (i == selected_device) {
                std::cout << " [SELECTED]";
            }
            
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    bool hasGPU() const {
        for (const auto& device : sycl_devices) {
            if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu) {
                return true;
            }
        }
        return false;
    }
    
    int getGPUCount() const {
        int count = 0;
        for (const auto& device : sycl_devices) {
            if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu) {
                count++;
            }
        }
        return count;
    }
};

// Updated initialization function
bool initialize(int device_id = 0) {
    std::lock_guard<std::mutex> lock(model_mutex);
    
    // Check for Intel GPU runtime
    std::cout << "Checking for Intel GPU support..." << std::endl;
    
    // Enumerate and select SYCL device
    sycl_manager.enumerateDevices();
    
    if (!sycl_manager.hasGPU()) {
        std::cerr << "ERROR: No Intel GPU detected!" << std::endl;
        std::cerr << "Please ensure:" << std::endl;
        std::cerr << "1. Intel GPU drivers are installed (intel-opencl-icd)" << std::endl;
        std::cerr << "2. Level Zero loader is installed (level-zero)" << std::endl;
        std::cerr << "3. Intel Compute Runtime is installed" << std::endl;
        std::cerr << "4. Run: clinfo or sycl-ls to verify GPU detection" << std::endl;
        return false;
    }
    
    std::cout << "Found " << sycl_manager.getGPUCount() << " Intel GPU(s)" << std::endl;
    
    sycl_manager.selectDevice(device_id);
    sycl_manager.printDevices();
    
    // Initialize llama backend
    llama_backend_init();
    
    // Continue with rest of initialization...
    // [rest of your initialization code]
}
