import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import utility
import transform


phn_dict = {}
phn_list = []
phn_count = 0

class Timit(Dataset):
    def __init__(self, annotations_file, data_dir, n_fft=512, transform=None, target_transform=None):
        df = pd.read_csv(annotations_file)
        df = df.sort_values('path_from_data_dir')
    
        self.df_wav = df[df['filename'].str.endswith('.WAV',na=False)]
        self.df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

        self.data_dir = data_dir
        self.n_fft = n_fft
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.df_wav.iat[idx, 5])
        sign, sr = sf.read(wav_path)
        spec = signal.stft(sign,sr,nperseg=self.n_fft)[2]

        phn_path = os.path.join(self.data_dir, self.df_phn.iat[idx, 5])
        df = pd.read_csv(phn_path, delimiter=' ', header=None)
        label = np.zeros(spec.shape[1])
        noverlap = self.n_fft//2

        global phn_dict
        global phn_list
        global phn_count

        for i in range(len(df)):
            begin = df.iat[i,0]//noverlap
            end = df.iat[i,1]//noverlap
            phn = df.iat[i,2]
            if not phn in phn_dict:
                phn_dict[phn] = phn_count
                phn_list.append(phn)
                phn_count += 1
            label[begin:end] = phn_dict[phn]
        if self.transform:
            spec = self.transform(spec)
        if self.target_transform:
            label = self.target_transform(label)
        
        return spec,label

class FramedTimit(Dataset):
    def __init__(self, annotations_file, npz_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.npz_dir = npz_dir
        self.transform = transform
        self.target_transform = target_transform
        self.cache_idx = None
        self.cache_npz = None

    def __len__(self):
        return self.annotations['max'].max()

    def __getitem__(self, idx):
        index = self.annotations.index[(self.annotations['min']<=idx)&(self.annotations['max']>idx)].tolist()[0]
        if self.cache_idx != index:
            npz_path = os.path.join(self.npz_dir, self.annotations.at[index, 'path'])
            self.cache_npz = np.load(npz_path)

        local_idx = idx - self.annotations.at[index, 'min']
        x = self.cache_npz['spec'][local_idx]
        y = self.cache_npz['label'][local_idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x,y

def load_timit():
    parser = argparse.ArgumentParser(description="load TIMIT and convert to npz file")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    n_fft = 512
    s2c = transform.Function(utility.spec2ceps)
    vtl = transform.VTL(n_fft,np.tanh(np.linspace(-0.5,0.5,32)))
    c2s = transform.Function(utility.ceps2spec)
    mel = transform.MelScale(n_fft,n_mels=64)
    trans = transform.Function(np.transpose)

    composed = transforms.Compose([s2c,vtl,c2s,mel,trans])

    train_data = Timit(os.path.join(args.path, 'train_data.csv'),
                       os.path.join(args.path, 'data/'),
                       n_fft=n_fft, transform=composed)   
    test_data = Timit(os.path.join(args.path, 'test_data.csv'),
                      os.path.join(args.path, 'data/'),
                      n_fft=n_fft, transform=composed)     

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    count = 0
    annotation = []
    max = 0
    for spec,label in train_dataloader:
        path = os.path.join(args.path, "data/npz/TRAIN/", f"data{count}")
        np.savez(path, spec=spec[0], label=label[0])
        annotation.append({'path':f"TRAIN/data{count}.npz",'min':max,'max':max+label.shape[1]})
        count += 1
        max += label.shape[1]

    df = pd.DataFrame(annotation)
    df.to_csv(os.path.join(args.path, 'train_npz.csv'))

    count = 0
    annotation = []
    max = 0
    for spec,label in test_dataloader:
        path = os.path.join(args.path, "data/npz/TEST/", f"data{count}")
        np.savez(path, spec=spec[0], label=label[0])
        annotation.append({'path':f"TEST/data{count}.npz",'min':max,'max':max+label.shape[1]})
        count += 1
        max += label.shape[1]

    df = pd.DataFrame(annotation)
    df.to_csv(os.path.join(args.path, 'test_npz.csv'))

def test_framedtimit():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = FramedTimit(os.path.join(args.path, 'train_npz.csv'),
                             os.path.join(args.path, 'data/npz/'))   
    test_data = FramedTimit(os.path.join(args.path, 'test_npz.csv'),
                            os.path.join(args.path, 'data/npz/'))

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    for batch, (X,y) in enumerate(train_dataloader):
        print(f"train batch = {batch}\r")

    for batch, (X,y) in enumerate(test_dataloader):
        print(f"test batch = {batch}\r")

if __name__=="__main__":
    test_framedtimit()