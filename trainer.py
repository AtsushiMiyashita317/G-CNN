import torch
from torch.utils.tensorboard import SummaryWriter

class IntIterator(object):
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        self.i += 1
        return i

def train(dataloader, model, loss_fn, optimizer, device, writer, iter):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).sum().item()/y.shape[0]

        i = iter.__next__()
        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/acc", acc, i)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} acc: {acc:>3f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device, writer):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    writer.add_scalar("test/loss", test_loss, 0)
    writer.add_scalar("test/acc", correct, 0)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def experiment(train_dataloader,test_dataloader,model,loss_fn,optimizer,device,epochs,log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    iter = IntIterator().__iter__()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device, writer, iter)
        
    test(test_dataloader, model, loss_fn, device, writer)
    print("Done!")