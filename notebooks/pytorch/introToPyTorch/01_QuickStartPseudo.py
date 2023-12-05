# 5개의 라이브러리를 로드

# 같은 라이브러리지만, 다른 방식을 이용해 데이터를 로드한다
## 학습용
## test용
# hint - tensor로 변환


# batch size를 정해 데이터를 쪼갠다


# 데이터 크기를 보자

# 디바이스를 설정해보자. mps 신기하다
# device 확인해보자

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss =loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss ,current = loss.item(), (batch +1) * len(X)
            print(f"loss : {loss:>7f} [{current:>5d}|/{size:>5d}]")

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, corret = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            corret += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    corret /= size
    print(f"Test Error : \n Accuracy : {(100*corret):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
   print(f"Epoch {t+1} \n -----------------")
   train(train_dataloader,model,loss_fn,optimizer)
   test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(),"model.pth")
print("Saved Pytorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

