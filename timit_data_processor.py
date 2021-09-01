import argparse
import os
import pickle

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

class TimitOld(Dataset):
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

class Timit(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir, 
                n_fft=512, transform1=None, transform2=None, target_transform=None, datasize=None):
        self.annotations = pd.read_csv(os.path.join(root, annotations_file))

        n_shift = n_fft//2

        self.annotations['n_frame'] = (self.annotations['length']+n_shift-1)//n_shift+1
        self.annotations['maxidx'] = self.annotations['n_frame'].cumsum()
        self.annotations['minidx'] = self.annotations['maxidx'].shift()
        self.annotations.at[self.annotations.index[0],'minidx'] = 0
        self.annotations['minidx'] = self.annotations['minidx'].astype(np.int64)

        with open(os.path.join(root, phncode_file),'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        
        self.data_dir = os.path.join(root, data_dir)
        self.n_fft = n_fft
        self.transform1 = transform1
        self.transform2 = transform2
        self.target_transform = target_transform
        self.cache_spec = None
        self.cache_label = None
        self.cache_range = (0,0)
        self.datasize = datasize

    def __len__(self):
        if self.datasize:
            return self.datasize
        else:
            return self.annotations['maxidx'].max()

    def __getitem__(self, idx):
        if not (self.cache_range[0]<=idx and idx<self.cache_range[1]):
            cand = self.annotations[self.annotations['maxidx']>idx]
            
            wav_path = os.path.join(self.data_dir, cand.iat[0, 1])
            sign, sr = sf.read(wav_path)
            self.cache_spec = signal.stft(sign,sr,nperseg=self.n_fft)[2]

            phn_path = os.path.join(self.data_dir, cand.iat[0, 2])
            df_phn = pd.read_csv(phn_path, delimiter=' ', header=None)
            self.cache_label = np.zeros(self.cache_spec.shape[1])
            noverlap = self.n_fft//2

            for i in range(len(df_phn)):
                begin = df_phn.iat[i,0]//noverlap
                end = df_phn.iat[i,1]//noverlap
                phn = df_phn.iat[i,2]
                assert phn in self.phn_dict
                    
                self.cache_label[begin:end] = self.phn_dict[phn]

            if self.transform1:
                self.cache_spec = self.transform1(self.cache_spec)    

            self.cache_range = (cand.iat[0, 6],cand.iat[0, 5])
        
        index = idx - self.cache_range[0]

        frame = self.cache_spec[...,index]
        label = self.cache_label[...,index]

        if self.transform2:
            frame = self.transform2(frame)
        if self.target_transform:
            label = self.target_transform(label)

        return frame, label

class TimitMetrics(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir):
        self.annotations = pd.read_csv(os.path.join(root, annotations_file))

        with open(os.path.join(root, phncode_file),'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        
        self.data_dir = os.path.join(root, data_dir)
        self.metrics = pd.DataFrame(index=self.phn_list, 
                                    columns=['count',
                                             'min_length',
                                             'max_length',
                                             'sum_length'],
                                    dtype=np.int)
        self.metrics.fillna(0,inplace=True)
        self.metrics.info()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        phn_path = os.path.join(self.data_dir, self.annotations.iat[idx, 2])
        df_phn = pd.read_csv(phn_path, delimiter=' ', header=None, names=['begin','end','code'])
        df_phn['length'] = df_phn['end'] - df_phn['begin']

        for code in self.phn_list:
            df_code = df_phn[df_phn['code']==code]
            self.metrics[code,'count'] += len(df_code)
            self.metrics[code,'min_length'] = min(self.metrics[code,'min_length'], df_code['length'].min(initial=np.inf))
            self.metrics[code,'max_length'] = max(self.metrics[code,'max_length'], df_code['length'].max(initial=0))
            self.metrics[code,'sum_length'] += df_code['length'].sum()

        return 0

class TimitTmp(Dataset):
    def __init__(self, annotations_file, data_dir):
        df = pd.read_csv(annotations_file)
        df = df.sort_values('path_from_data_dir')
    
        self.df_wav = df[df['filename'].str.endswith('.WAV',na=False)]
        self.df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

        self.data_dir = data_dir

        self.annotations_new = []

    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.df_wav.iat[idx, 5])
        sign, sr = sf.read(wav_path)

        phn_path = os.path.join(self.data_dir, self.df_phn.iat[idx, 5])
        df = pd.read_csv(phn_path, delimiter=' ', header=None)

        global phn_dict
        global phn_list
        global phn_count

        for i in range(len(df)):
            phn = df.iat[i,2]
            if not phn in phn_dict:
                phn_dict[phn] = phn_count
                phn_list.append(phn)
                phn_count += 1

        self.annotations_new.append({'wav_path':self.df_wav.iat[idx, 5],
                                     'phn_path':self.df_phn.iat[idx, 5],
                                     'length':sign.shape[0]})
        
            
        return sign

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

def main():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    n_fft = 512
    s2c = transform.Function(utility.spec2ceps)
    vtl = transform.VTL(n_fft,np.tanh(np.linspace(-0.5,0.5,32)))
    c2s = transform.Function(utility.ceps2spec)
    mel = transform.MelScale(n_fft,n_mels=64)
    trans = transform.Function(np.transpose)
    addc = transform.Function(np.expand_dims, axis=0)

    composed1 = transforms.Compose([s2c,vtl,c2s,mel])
    composed2 = transforms.Compose([trans,addc])

    train_data = Timit(args.path,'train_annotations.csv','phn.pickle','data/',n_fft=n_fft,transform1=composed1,transform2=addc)
    test_data = Timit(args.path,'test_annotations.csv','phn.pickle','data/',n_fft=n_fft,transform1=composed1,transform2=addc)

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    for batch, (X,y) in enumerate(train_dataloader):
        print(f"train batch = {batch}\r")

    for batch, (X,y) in enumerate(test_dataloader):
        print(f"test batch = {batch}\r")

def tmp():
    parser = argparse.ArgumentParser(description="load TIMIT and convert to npz file")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = TimitTmp(os.path.join(args.path, 'train_data.csv'),
                       os.path.join(args.path, 'data/'))   
    test_data = TimitTmp(os.path.join(args.path, 'test_data.csv'),
                      os.path.join(args.path, 'data/'))  

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    count = 0
    
    for sign in train_dataloader:
        print(f"processing train... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(train_data.annotations_new)
    df.to_csv(os.path.join(args.path, 'train_annotations.csv'))


    with open("phn.pickle", "wb") as f:
        pickle.dump((phn_dict,phn_list,phn_count), f)

    print('\n')

    for sign in test_dataloader:
        print(f"processing test... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(test_data.annotations_new)
    df.to_csv(os.path.join(args.path, 'test_annotations.csv'))

def metrics():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = TimitMetrics(args.path,'train_annotations.csv','phn.pickle','data/')
    test_data = TimitMetrics(args.path,'test_annotations.csv','phn.pickle','data/')

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    for batch, x in enumerate(train_dataloader):
        print(f"train batch = {batch}\r")

    for batch, x in enumerate(test_dataloader):
        print(f"test batch = {batch}\r")

if __name__=="__main__":
    metrics()