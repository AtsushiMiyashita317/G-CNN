import argparse

import librosa
import numpy as np
from scipy import signal
import soundfile
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms

import utility
from TIMIT.timit_data_processor import Timit
import MyUtility.mydataset


def filtfilt(x,a):
    """
        Apply IIR filter (z^(-1)-a)/(1-az^(-1))
        # Args
            x (ndarray, axis=(time,...)):
                input signal
            a (float or ndarray, axis=(...,)):
                warping parameter
        # Returns
            y (ndarray, axis=(time,...)):
                output signal
        # Note
            x.shape[1:] and a.shape must be same.

    """
    y = a*x
    y[1:] += x[:-1]
    for i in range(1,y.shape[0]):
        y[i] -= a*y[i-1]
    return y

def vtl_mat(dim, a):
    """
        Create transform matrix for VTL.
        # Args
            dim (int): dimension of matrix
            a (float or ndarray, shape=(k,)):
                warping parameter
        # Returns
            mat (ndarray, shape=((k,)dim,dim)):
                transform matrix
        # Examples
            >>> dim = ceps.shape[0]
            >>> ceps_vtlt = vtl_mat(dim,a) @ ceps
    """
    if type(a) is float:
        mat = np.zeros((dim,dim))
    else:
        mat = np.zeros((dim,dim,a.size))
    
    mat[0,0] = 1

    for i in range(1,dim):
        mat[i] = filtfilt(mat[i-1],a)

    mat = np.transpose(mat)
    return mat

class VTL(object):
    """
        VTL transformer
    """
    def __init__(self, a, n_fft=1024):
        """
            VTL parameter setting
            # Args
                half (int): dimension of cepstrum along quefrency axis
                a (float or ndarray, shape=(k,)):
                    warping parameter
        """
        assert np.all(np.abs(a) < 1)
        self.n_fft = n_fft
        self.a = a
        self.mat = vtl_mat(n_fft//2+1,a)

    def __call__(self, sign):
        """
            Compose transform
            # Args
                spec (ndarray, axis=(...,freq,time)):
                    input spectrum
            # Returns
                spec_trans (ndarray, axis=((k,)...,freq,time)):
                    transformed spectrum
        """
        # commentout after test
        # assert self.dim == input.shape[0]
        
        spec = signal.stft(sign, nperseg=self.n_fft)[2]
        ceps = utility.spec2ceps(spec)
        
        ceps_trans = np.zeros(self.a.shape + ceps.shape)
        dim = self.n_fft//2+1
        
        positive = ceps[:dim]
        negative = np.roll(ceps,-1,axis=0)[-1:-dim-1:-1]
        ceps_trans[:,:dim] = self.mat @ positive
        ceps_trans[:,-1:1-dim:-1] = (self.mat @ negative)[:,1:dim-1]

        spec_trans = utility.ceps2spec(ceps_trans)
        sign_trans = signal.istft(spec_trans, nperseg=self.n_fft)[1]

        return sign_trans

class VTL_Invariant(object):
    """
        Extract VTL_Invariant
    """
    def __init__(self, n_fft, dropphase=False):
        """
            VTL parameter setting
            # Args
                half (int): dimension of cepstrum along quefrency axis
                a (float or ndarray, shape=(k,)):
                    warping parameter
        """
        self.dim = n_fft//2 + 1
        self.dropphase = dropphase

        delem = np.concatenate([np.arange(1,self.dim),
                                np.arange(n_fft-self.dim-1,-1,-1)])
        upper = np.diag(delem,1)
        delem = np.concatenate([np.arange(self.dim-1),
                                np.arange(n_fft-self.dim,0,-1)])
        lower = np.diag(delem,-1)
        sum = upper - lower
        self.l,self.v = np.linalg.eig(sum)
        self.v_inv = np.linalg.inv(self.v)

    def __call__(self, spec):
        """
            Compose transform
            # Args
                spec (ndarray, axis=(...,freq,time)):
                    input spectrum
            # Returns
                spec_trans (ndarray, axis=((k,)...,freq,time)):
                    transformed spectrum
        """
    
        ceps = utility.spec2ceps(spec, self.dropphase)

        ivar = self.v @ np.abs(self.v_inv @ ceps)

        spec_like = utility.ceps2spec(ivar)

        return spec_like

class Normalize(object):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, input):
        mean = np.mean(input,self.axis,keepdims=True)
        std = np.std(input,self.axis,keepdims=True)
        return (input-mean)/std

class MelScale(object):
    def __init__(self, n_fft, sr=16000, n_mels=128):
        self.fb = librosa.filters.mel(sr,n_fft,n_mels=n_mels)

    def __call__(self, input):
        output = self.fb @ input
        return output

class Function(object):
    def __init__(self, f, idx_ret=None, *args, **kwargs):
        self.f = f
        self.idx_ret = idx_ret
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input):
        output = self.f(input,*self.args,**self.kwargs)
        if self.idx_ret:
            return output[self.idx_ret]
        else:
            return output
    

def test_vtl():
    # process args
    parser = argparse.ArgumentParser(description="test class VTL")
    parser.add_argument('sc', type=str, help="input filename with extension .wav")
    args = parser.parse_args()

    n_fft = 1024
    sign, sr = soundfile.read(args.sc)
    
    a = np.tanh(np.linspace(-0.3,0.3,9))
    vtl = VTL(a, n_fft)

    sign_trans = vtl(sign)

    # spec_trans = signal.stft(sign_trans,nperseg=256)[2]
    # sign_trans = signal.istft(spec_trans,nperseg=256)[1]


    for i in range(a.size):
        soundfile.write(f"result/test_vtl/test{i}.wav",sign_trans[i],sr)


def main():
    # process args
    parser = argparse.ArgumentParser(description="transform data")
    parser.add_argument('sc', type=str, help="input filename with extension .npz")
    parser.add_argument('dst', type=str, help="output filename with extension .npz")
    args = parser.parse_args()

    npz = np.load(args.sc)
    x_train = npz['x_train'][:,:10000]
    y_train = npz['y_train'][:10000]
    x_test = npz['x_test'][:,:1000]
    y_test = npz['y_test'][:1000]
    
    s2c = Function(utility.spec2ceps)
    vtl = VTL(x_train.shape[0],np.tanh(np.linspace(-0.3,0.3)))
    composed = transforms.Compose([s2c,vtl])

    x_train_transformed = composed(x_train)
    x_test_transformed = composed(x_test)

    print(x_train_transformed.shape)
    print(x_test_transformed.shape)
    
    np.savez(args.dst,x_train=x_train_transformed,y_train=y_train,x_test=x_test_transformed,y_test=y_test)

def transform():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("sc", type=str, help="path to the directory that has annotation files")
    parser.add_argument("dst", type=str, help="path to the directory that transformed data is saved to")
    args = parser.parse_args()

    n_fft =  256
    vtl = VTL(np.tanh(np.linspace(-0.3,0.3,9)),n_fft)
    mel = MelScale(n_fft,n_mels=40)
    trans = Function(np.transpose)

    train_data = Timit(args.sc,'TIMIT/train_annotations.csv','TIMIT/phn.pickle','data/',
                       n_fft=n_fft,signal_transform=vtl,spec_transform=mel,frame_transform=trans)
    test_data = Timit(args.sc,'TIMIT/test_annotations.csv','TIMIT/phn.pickle','data/',
                      n_fft=n_fft,signal_transform=vtl,spec_transform=mel,frame_transform=trans)

    MyUtility.mydataset.save(train_data, args.dst, "TRAIN", 128)
    MyUtility.mydataset.save(test_data, args.dst, "TEST", 128)

def load():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("sc", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = MyUtility.mydataset.MyDataset(args.sc, 'TRAIN')
    test_data = MyUtility.mydataset.MyDataset(args.sc, 'TEST')

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    print(train_data[0][0].shape)

    for batch, (X,y) in enumerate(train_dataloader):
        print(f"processing train... batch = {batch}\r",end='')

    print()

    for batch, (X,y) in enumerate(test_dataloader):
        print(f"processing test... batch = {batch}\r",end='')


if __name__ == "__main__":
    transform()