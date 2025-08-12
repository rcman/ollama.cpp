#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <string>

class IntelGPULevelZero {
private:
ze_driver_handle_t driver;
ze_device_handle_t device;
ze_context_handle_t context;
ze_command_queue_handle_t cmdQueue;
ze_command_list_handle_t cmdList;

public:
bool initialize() {
// Initialize Level Zero
if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
std::cerr << “Failed to initialize Level Zero\n”;
return false;
}

```
    // Get driver
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);
    if (driverCount == 0) {
        std::cerr << "No Level Zero drivers found\n";
        return false;
    }
    
    std::vector<ze_driver_handle_t> drivers(driverCount);
    zeDriverGet(&driverCount, drivers.data());
    driver = drivers[0];
    
    // Get device
    uint32_t deviceCount = 0;
    zeDeviceGet(driver, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "No devices found\n";
        return false;
    }
    
    std::vector<ze_device_handle_t> devices(deviceCount);
    zeDeviceGet(driver, &deviceCount, devices.data());
    device = devices[0];
    
    // Create context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    if (zeContextCreate(driver, &contextDesc, &context) != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to create context\n";
        return false;
    }
    
    // Create command queue
    ze_command_queue_desc_t queueDesc = {};
    queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queueDesc.ordinal = 0;
    queueDesc.index = 0;
    queueDesc.flags = 0;
    queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    
    if (zeCommandQueueCreate(context, device, &queueDesc, &cmdQueue) != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to create command queue\n";
        return false;
    }
    
    // Create command list
    ze_command_list_desc_t listDesc = {};
    listDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    listDesc.commandQueueGroupOrdinal = 0;
    
    if (zeCommandListCreate(context, device, &listDesc, &cmdList) != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to create command list\n";
        return false;
    }
    
    return true;
}

void parallelCompute(std::vector<float>& data, float multiplier) {
    size_t dataSize = data.size() * sizeof(float);
    
    // Allocate device memory
    ze_device_mem_alloc_desc_t deviceDesc = {};
    deviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    deviceDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
    deviceDesc.ordinal = 0;
    
    void* deviceBuffer;
    if (zeMemAllocDevice(context, &deviceDesc, dataSize, 64, device, &deviceBuffer) != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to allocate device memory\n";
        return;
    }
    
    // Copy data to device
    zeCommandListAppendMemoryCopy(cmdList, deviceBuffer, data.data(), dataSize, nullptr, 0, nullptr);
    
    // SPIR-V kernel for simple multiplication (simplified example)
    const char* kernelSource = R"(
        kernel void multiply_kernel(global float* data, float multiplier, int count) {
            int gid = get_global_id(0);
            if (gid < count) {
                data[gid] *= multiplier;
            }
        }
    )";
    
    // Note: In real implementation, you'd compile OpenCL to SPIR-V
    // This is a simplified example showing the Level Zero workflow
    
    // Execute command list
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
    zeCommandQueueSynchronize(cmdQueue, UINT64_MAX);
    
    // Copy result back
    zeCommandListReset(cmdList);
    zeCommandListAppendMemoryCopy(cmdList, data.data(), deviceBuffer, dataSize, nullptr, 0, nullptr);
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
    zeCommandQueueSynchronize(cmdQueue, UINT64_MAX);
    
    // Cleanup
    zeMemFree(context, deviceBuffer);
}

void getDeviceInfo() {
    ze_device_properties_t deviceProps = {};
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(device, &deviceProps);
    
    std::cout << "Device Name: " << deviceProps.name << std::endl;
    std::cout << "Max Work Group Size: " << deviceProps.maxTotalGroupSize << std::endl;
    std::cout << "Max Memory Alloc Size: " << deviceProps.maxMemAllocSize << " bytes" << std::endl;
    
    ze_device_compute_properties_t computeProps = {};
    computeProps.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    zeDeviceGetComputeProperties(device, &computeProps);
    
    std::cout << "Max Group Count X: " << computeProps.maxGroupCountX << std::endl;
    std::cout << "Max Group Size X: " << computeProps.maxGroupSizeX << std::endl;
}

~IntelGPULevelZero() {
    if (cmdList) zeCommandListDestroy(cmdList);
    if (cmdQueue) zeCommandQueueDestroy(cmdQueue);
    if (context) zeContextDestroy(context);
}
```

};

int main() {
IntelGPULevelZero gpu;
if (!gpu.initialize()) {
return -1;
}

```
gpu.getDeviceInfo();

std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
gpu.parallelCompute(data, 2.0f);

std::cout << "Results: ";
for (float val : data) {
    std::cout << val << " ";
}
std::cout << std::endl;

return 0;
```

}