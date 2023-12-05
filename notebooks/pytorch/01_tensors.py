import torch
import numpy as np

# Initialize tensor
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(f"x_data : {x_data}")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np:{x_np}")

# Tensor from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n{x_ones} \n")
x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Tensor Attributes
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatyp of tensor: {tensor.dtype}")
print(f"Deevice tensor is stored on : {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on : {tensor.device}")

tensor = torch.ones(4,4)
tensor[:,1]=0
print(tensor)

# joining tensors. torch.stack도 있지ㄴ 조ㅁ 다다다
t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(f"Horizontal stacked tensors : \n {t1} \n")

# elemtwise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor*tensor \n {tensor * tensor} \n")

# dot product
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n")
