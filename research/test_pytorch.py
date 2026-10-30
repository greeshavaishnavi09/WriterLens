import torch
import torchvision

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)

# STEP 1: Create tensors
x = torch.tensor(
    [[1, 2], [3, 4]],
    dtype=torch.float32
)

y = torch.tensor(
    [[5, 6], [7, 8]],
    dtype=torch.float32
)

# STEP 2: Matrix multiplication
result = torch.matmul(x, y)

print("\nTensor X:")
print(x)

print("\nTensor Y:")
print(y)

print("\nMatrix Multiplication:")
print(result)

# STEP 3: Check shape
print("\nShape:", result.shape)

# STEP 4: Check device
print("Device:", result.device)

# STEP 5: Automatic differentiation
a = torch.tensor(2.0, requires_grad=True)

b = a ** 2 + 3 * a + 1

b.backward()

print("\nCalculated Gradient:", a.grad)

print("\nCUDA Available:", torch.cuda.is_available())