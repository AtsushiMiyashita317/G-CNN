import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import trainer
import transform
from MyUtility.mydataset import MyDataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(64,32,(8,3)),
            nn.ReLU(),
            nn.MaxPool2d((3,7), stride=(3,7)),
            nn.Flatten(),
            nn.Linear(32*6*1,1024),
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

    epochs = 100
    exp_dir = f"./logs/{args.exname}"

    trainer.experiment(train_dataloader,test_dataloader,model,loss_fn,optimizer,device,epochs,exp_dir)


if __name__=="__main__":
    main()