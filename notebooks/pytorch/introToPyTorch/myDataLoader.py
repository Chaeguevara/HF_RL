import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform = ToTensor()
        )

test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform = ToTensor()
        )


col,row = 3,3
figure = plt.figure(figsize=(8,8))


for i in range(1,col*row+1):
    idx = torch.randint(9,(1,)).item()
    img, label = training_data[idx]
    figure.add_subplot(row,col,i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(0),cmap="gray")
plt.show()




