import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import transform
from MyUtility.mydataset import MyDataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(40,10,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,7), stride=(2,7)),
            nn.Flatten(),
            nn.Linear(10*6*1,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,61),
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits

def main():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    parser.add_argument("exname", type=str, help="the name of experiment")
    parser.add_argument("--model_path", type=str, help="the name of model file")
    args = parser.parse_args()

    trans = transform.Function(np.transpose, axes=(1,0,2))

    train_data = MyDataset(args.path, 'TRAIN', transform=trans)
    test_data = MyDataset(args.path, 'TEST', transform=trans)

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if args.model_path:
        model = Model().to(device)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    else:
        model = Model().to(device)
        print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 10000
    size = len(train_dataloader.dataset)

    writer = SummaryWriter(log_dir=f"./logs/{args.exname}")
    i = 0

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
            
            # 損失誤差を計算
            pred = model(X)
            loss = loss_fn(pred, y)
            acc = (pred.argmax(1) == y).type(torch.float).sum().item()/y.shape[0]
            
            writer.add_scalar("training/loss", loss, i)
            writer.add_scalar("training/accuracy", acc, i)
            i += 1

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} acc: {acc:>3f} [{current:>5d}/{size:>5d}]")

        if t % 10 == 0:
            model_path = f"./logs/{args.exname}/model{t}.pth"
            torch.save(model, model_path)

    size = len(test_dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    print("Done!")

    model_path = f"./logs/{args.exname}/model.pth"
    torch.save(model, model_path)

    writer.close()


if __name__=="__main__":
    main()