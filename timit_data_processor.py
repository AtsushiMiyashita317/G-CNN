import argparse
import os

import numpy as np
import pickle
import pandas as pd
import soundfile as sf
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


phn_dict = {}
phn_list = []
phn_count = 0

def feature_extraction(path,csvname,fwfdst):
    """
    """
    df = pd.read_csv(path+csvname)
    
    df = df.sort_values('path_from_data_dir')
    
    df_wav = df[df['filename'].str.endswith('.WAV',na=False)]
    df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

    x_df_wav = df_wav['path_from_data_dir'].values
    x_df_phn = df_phn['path_from_data_dir'].values
    assert len(x_df_wav) == len(x_df_phn)

    n_fft = 512
    noverlap = n_fft//2
    x = np.zeros((n_fft//2+1,0))
    y = np.zeros(0)

    fwf = []

    global phn_dict
    global phn_list
    global phn_count
        
    for i in range(len(x_df_wav)):
        name_wav = x_df_wav[i].replace('.WAV','')
        name_phn = x_df_phn[i].replace('.PHN','')
        assert name_wav == name_phn

        sign, sr = sf.read(f"{path}data/{x_df_wav[i]}")
        spec = signal.stft(sign,sr,nperseg=n_fft)[2]

        df = pd.read_csv(f"{path}data/{x_df_phn[i]}",delimiter=' ',header=None)
        label = np.zeros(spec.shape[1])
        for i in range(len(df)):
            begin = df.iat[i,0]//noverlap
            end = df.iat[i,1]//noverlap
            phn = df.iat[i,2]
            if not phn in phn_dict:
                phn_dict[phn] = phn_count
                phn_list.append(phn)
                phn_count += 1
            label[begin:end] = phn_dict[phn]

        begin = y.shape[0]
        x = np.concatenate([x,spec],axis=1)
        y = np.concatenate([y,label],axis=0)
        end = y.shape[0]
        fwf.append({'begin':begin,'end':end,'file':x_df_wav[i]})
    
    df = pd.DataFrame(fwf)
    df.to_csv(fwfdst)
    return x,y

class Timit(Dataset):
    def __init__(self, annotations_file, data_dir, n_fft=512):
        df = pd.read_csv(annotations_file)
        df = df.sort_values('path_from_data_dir')
    
        self.df_wav = df[df['filename'].str.endswith('.WAV',na=False)]
        self.df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

        self.data_dir = data_dir
        self.n_fft = n_fft

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

        return spec,label

class FramedTimit(Dataset):
    def __init__(self, annotations_file, npz_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.npz_dir = npz_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.annotations['max'].max()

    def __getitem__(self, idx):
        df = self.annotations[(self.annotations['min']<=idx)&(self.annotations['max']>idx)]
        npz_path = os.path.join(self.npz_dir, df.iat[0, 0])
        local_idx = idx - df.iat[0, 1]
        npz = np.load(npz_path)
        x = npz['x'][local_idx]
        y = npz['y'][local_idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y

def main():
    parser = argparse.ArgumentParser(description="process timit data")
    parser.add_argument("path", type=str, help="path to the directory that has data-path files")
    parser.add_argument("dst", type=str, help="feature is saved as the file")
    args = parser.parse_args()

    x_train,y_train = feature_extraction(args.path,"train_data.csv","train_fwf.csv")
    x_test,y_test = feature_extraction(args.path,"test_data.csv","test_fwf.csv")

    np.savez(args.dst,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

    f = open('phn.pickle', 'wb')
    pickle.dump(phn_list, f)
    
def load_timit():
    parser = argparse.ArgumentParser(description="load TIMIT and convert to npz file")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = Timit(os.path.join(args.path, 'train_data.csv'),
                       os.path.join(args.path, 'data/'))   
    test_data = Timit(os.path.join(args.path, 'test_data.csv'),
                      os.path.join(args.path, 'data/'))     

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    count = 0
    annotation = []
    max = 0
    for spec,label in train_dataloader:
        path = os.path.join(args.path, "data/npz/TRAIN/", f"data{count}")
        np.savez(path, spec=spec, label=label)
        annotation.append({'path':f"{path}.npz",'min':max,'max':max+label.shape[0]})
        count += 1
        max += label.size

    df = pd.DataFrame(annotation)
    df.to_csv(os.path.join(args.path, 'train_npz.csv'))

    count = 0
    annotation = []
    max = 0
    for spec,label in test_dataloader:
        path = os.path.join(args.path, "data/npz/TEST/", f"data{count}")
        np.savez(path, spec=spec, label=label)
        annotation.append({'path':f"{path}.npz",'min':max,'max':max+label.shape[0]})
        count += 1
        max += label.size

    df = pd.DataFrame(annotation)
    df.to_csv(os.path.join(args.path, 'test_npz.csv'))


if __name__=="__main__":
    load_timit()