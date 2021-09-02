import argparse
import os

import numpy as np
import torch
from torch import nn
from torch._C import dtype
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import transform
import timit_data_processor
import utility


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(15,91,(8,3)),
            nn.ReLU(),
            nn.MaxPool2d((6,7), stride=(6,7)),
            nn.Flatten(),
            nn.Linear(91*5*1,1024),
            nn.ReLU(),
            nn.Linear(1024,61),
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).sum().item()/y.shape[0]
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} acc: {acc:>3f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    n_fft = 256
    vtl = transform.VTL(n_fft,np.tanh(np.linspace(-0.5,0.5,9)),dropphase=True)
    mel = transform.MelScale(n_fft,n_mels=40)
    trans = transform.Function(np.transpose)


    composed1 = transforms.Compose([vtl,mel])

    train_data = timit_data_processor.Timit(args.path,'train_annotations.csv','phn.pickle','data/',
                                            n_fft=n_fft,transform1=composed1,transform2=trans,datasize=512)
    test_data = timit_data_processor.Timit(args.path,'test_annotations.csv','phn.pickle','data/',
                                           n_fft=n_fft,transform1=composed1,transform2=trans,datasize=1024)

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = Model().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        # test(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__=="__main__":
    main()