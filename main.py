import argparse
import os

import torch
from torch import nn
from torch.nn.modules import linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timit_data_processor


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16,16,3),
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(13*5,64),
            nn.ReLU(),
            nn.Linear(64,61)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits


def main():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = timit_data_processor.FramedTimit(
        os.path.join(args.path, 'train_npz.csv'),
        os.path.join(args.path, 'data/npz/')
        )   
    test_data = timit_data_processor.FramedTimit(
        os.path.join(args.path, 'test_npz.csv'),
        os.path.join(args.path, 'data/npz/')
        )

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = Model().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__=="__main__":
    main()