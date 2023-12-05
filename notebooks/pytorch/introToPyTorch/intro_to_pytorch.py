import torch

# tensor manipulation
z = torch.zeros(5,3)

print(z)
print(z.dtype)

i = torch.ones((5,3), dtype=torch.int16)
print(i)

torch.manual_seed(1729)
r1 = torch.rand(2,2)
print("A random tensor r1:")
print(r1)

r2 = torch.rand(2,2)
print(f'\n A diffrent random tensor:{r2}')

torch.manual_seed(1729)
r3 = torch.rand(2,2)
print(f"\n Should match r1 : \n{r3}")

ones = torch.ones(2,3)
print(f"ones : {ones}")

twos = torch.ones(2,3) * 2
print(f"twos : {twos}")

threes = ones + twos
print(f"threes: {threes}")
print(f"threes.shape: {threes.shape}")

r = (torch.rand(2,2) -0.5)*2
print(f"A random matrix, r: \n{r}")

print(f'\n Absoulte value of r: \n {r}')

print(f'\n Inverse sine of r: \n{torch.asin(r)}')

print(f"Determinant of r: \n{torch.det(r)}")
print(f"Singular value decomposition of r: \n{torch.svd(r)}")

print(f"Average and standrad deviation of r: \n{torch.std_mean(r)}")
print(f"Maximum value of r:\n{torch.max(r)}")


