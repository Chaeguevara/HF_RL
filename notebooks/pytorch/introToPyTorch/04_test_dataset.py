import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        )

test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
        )

cols, rows = 3,3

figure = plt.figure(figsize=(8,8))

for i in range(1,cols*rows+1):
    idx = torch.randint(10,(1,)).item()
    img,label = train_data[idx]
    figure.add_subplot(rows,cols,i)
    plt.title(idx)
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap='gray')

plt.show()




