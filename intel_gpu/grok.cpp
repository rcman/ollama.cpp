import torch

# Check if Intel GPU (XPU) is available
if torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Intel GPU (XPU)")
else:
    device = torch.device("cpu")
    print("Intel GPU not available; falling back to CPU")

# Create a tensor and move it to the device
tensor_a = torch.tensor([1.0, 2.0, 3.0]).to(device)
tensor_b = torch.tensor([4.0, 5.0, 6.0]).to(device)

# Perform a simple operation (element-wise multiplication)
result = tensor_a * tensor_b

print("Result:", result)