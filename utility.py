import numpy as np

def spec2ceps(spec):
    spec_db = np.log(spec)
    ceps = np.fft.ifft(spec_db,axis=-2)
    return ceps.real


def ceps2spec(ceps):
    spec_db = np.fft.rfft(ceps,axis=-2)
    spec = np.exp(spec_db)
    return spec