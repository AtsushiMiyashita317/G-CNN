import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import trainer
import transform
from TIMIT.timit_data_processor import Timit

class FullConect(nn.Module):
    def __init__(self, in_features,out_features,n_hiddenlayers,hidden_channel=1024):
        super(FullConect, self).__init__()
        self.flatten = nn.Flatten()
        self.inputlayer = nn.Linear(in_features,hidden_channel)
        self.hiddenlayer = nn.Linear(hidden_channel,hidden_channel)
        self.outputlayer = nn.Linear(hidden_channel,out_features)
        self.activation = nn.ReLU()
        self.n_hiddenlayers = n_hiddenlayers

    def forward(self, input):
        x = self.flatten(input)
        x = self.inputlayer(x)
        x = self.activation(x)
        for i in range(self.n_hiddenlayers):
            x = self.hiddenlayer(x)
            x = self.activation(x)
        x = self.outputlayer(x)
        logits = self.activation(x)
        return logits


def feature_test(train_dataloader, test_dataloader, n_class, max_hiddenlayers, epochs, log_dir, start_hiddenlayers=3):
    train_data = train_dataloader.__iter__().next()
    in_features = torch.numel(train_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    for n_hiddenlayers in range(start_hiddenlayers,max_hiddenlayers):
        model = FullConect(in_features,n_class,n_hiddenlayers).to(device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        exp_dir = os.path.join(log_dir, f"hiddenlayers{n_hiddenlayers}")

        trainer.experiment(train_dataloader,test_dataloader,model,loss_fn,optimizer,device,epochs,exp_dir)


def main():
    parser = argparse.ArgumentParser(description="test feature")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    n_ffts = [256,512,1024]
    n_frames = [15,30,60]
    n_melss = [40,80,160]

    for n_fft in n_ffts:
        for n_frame in n_frames:
            for n_mels in n_melss:
                log_dir = f"./logs/feature_test/n_fft{n_fft}/n_frame{n_frame}/n_mels{n_mels}/"

                mel = transform.MelScale(n_fft,n_mels=n_mels)

                train_data = Timit(args.path,'TIMIT/train_annotations.csv','TIMIT/phn.pickle','data/',
                                n_fft=n_fft,n_frame=n_frame,spec_transform=mel)
                test_data = Timit(args.path,'TIMIT/test_annotations.csv','TIMIT/phn.pickle','data/',
                                n_fft=n_fft,n_frame=n_frame,spec_transform=mel)

                train_dataloader = DataLoader(train_data, batch_size=128)
                test_dataloader = DataLoader(test_data, batch_size=128)

                feature_test(train_dataloader,test_dataloader,61,8,10,log_dir)

if __name__ == "__main__":
    main()