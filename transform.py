import argparse

import numpy as np
from scipy import signal
import soundfile

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
        
        half = dim//2+1
        filt = np.zeros((half,a.size))
        mat[0,0] = 1
        filt[0] = 1
        for i in range(1,half):
            filt = VTL.filtfilt(filt,a)
            mat[i,:half] = filt
            mat[-i,0] = filt[0]
            mat[-i,:-half:-1] = filt[1:]

        mat = np.transpose(mat)
        return mat

    def __init__(self, dim, a):
        """
            VTL parameter setting
            # Args
                dim (int): dimension of cepstrum along quefrency axis
                a (float or ndarray, shape=(k,)):
                    warping parameter
        """
        assert np.all(np.abs(a) < 1)
        self.dim = dim
        self.a = a
        self.mat = VTL.vtl_mat(dim,a)

    def __call__(self, input):
        """
            Compose transform
            # Args
                ceps (ndarray, axis=(...,quef,frame)):
                    input cepstrum
            # Returns
                ceps_trans (ndarray, axis=((k,)...,quef,frame)):
                    transformed cepstrum
        """
        # commentout after test
        # assert self.dim == input.shape[0]

        output = self.mat @ input
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
    ceps = utility.spec2ceps(spec)
    spec = utility.ceps2spec(ceps)
    sign = signal.istft(spec,nperseg=n_fft)[1]
    soundfile.write(f"result/test_vtl/test.wav",sign,sr)

    dim = ceps.shape[0]
    a = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    vtl = VTL(dim, a)
    assert vtl.mat.shape == (a.size,dim,dim)

    ceps_trans = vtl(ceps)
    assert ceps_trans.shape == a.shape + ceps.shape

    spec_trans = utility.ceps2spec(ceps_trans)
    for i in range(a.size):
        sign_trans = signal.istft(spec_trans[i],nperseg=n_fft)[1]
        soundfile.write(f"result/test_vtl/test{i}.wav",sign_trans,sr)


if __name__ == "__main__":
    test_vtl()