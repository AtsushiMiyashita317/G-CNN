import argparse

import librosa
import numpy as np
from numpy.core.function_base import linspace
from scipy import signal
import soundfile
import torch
from torchvision import transforms

import utility


class VTL(object):
    """
        VTL transformer
    """
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
            mat[i] = VTL.filtfilt(mat[i-1],a)

        mat = np.transpose(mat)
        return mat

    def __init__(self, n_fft, a, dropphase=False):
        """
            VTL parameter setting
            # Args
                half (int): dimension of cepstrum along quefrency axis
                a (float or ndarray, shape=(k,)):
                    warping parameter
        """
        assert np.all(np.abs(a) < 1)
        self.dim = n_fft//2 + 1
        self.a = a
        self.dropphase = dropphase
        self.mat = VTL.vtl_mat(self.dim,a)

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
        # commentout after test
        # assert self.dim == input.shape[0]

        ceps = utility.spec2ceps(spec, self.dropphase)

        ceps_trans = np.zeros(self.a.shape + ceps.shape)
        if self.dropphase:
            ceps_trans[:,:self.dim] = self.mat @ ceps[:self.dim]
            ceps_trans[:,-1:1-self.dim:-1] = ceps_trans[:,1:self.dim-1]
        else:
            positive = ceps[:self.dim]
            negative = np.roll(ceps,-1,axis=0)[-1:-self.dim-1:-1]
            ceps_trans[:,:self.dim] = self.mat @ positive
            ceps_trans[:,-1:1-self.dim:-1] = (self.mat @ negative)[:,1:self.dim-1]

        spec_trans = utility.ceps2spec(ceps_trans)

        if self.dropphase:
            spec_trans = np.abs(spec_trans)

        return spec_trans

class MelScale(object):
    def __init__(self, n_fft, sr=16000, n_mels=128):
        self.fb = librosa.filters.mel(sr,n_fft,n_mels=n_mels)

    def __call__(self, input):
        output = self.fb @ input
        return output

class Function(object):
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input):
        output = self.f(input,*self.args,**self.kwargs)
        return output
    

def test_vtl():
    # process args
    parser = argparse.ArgumentParser(description="test class VTL")
    parser.add_argument('sc', type=str, help="input filename with extension .wav")
    args = parser.parse_args()

    n_fft = 1024
    sign, sr = soundfile.read(args.sc)
    spec = signal.stft(sign,sr,nperseg=n_fft)[2]
    
    dim = n_fft//2 + 1
    a = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    vtl = VTL(n_fft, a, dropphase=True)
    assert vtl.mat.shape == (a.size,dim,dim)

    spec_trans = vtl(spec)
    assert spec_trans.shape == a.shape + spec.shape

    for i in range(a.size):
        sign_trans = signal.istft(spec_trans[i],nperseg=n_fft)[1]
        soundfile.write(f"result/test_vtl/test{i}.wav",sign_trans,sr)


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


if __name__ == "__main__":
    test_vtl()