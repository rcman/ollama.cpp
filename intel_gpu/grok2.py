import torch
import torchvision.models as models

# Check for XPU availability
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained model
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
model = model.to(device)

# Create random input data (batch size 1, 3 channels, 224x224 image)
data = torch.rand(1, 3, 224, 224).to(device)

# Run inference
with torch.no_grad():
    output = model(data)

print("Inference completed. Output shape:", output.shape)