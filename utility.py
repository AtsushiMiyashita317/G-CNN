import numpy as np

def spec2ceps(spec, dropphase=False):
    """
        Get cepstrum from spectrum.
        # Args
            spec (ndarray, axis=(freq,time)):
                input specturm (one sided, complex). 
            dropphase (bool):
                If True, phase is ignored.
        # Returns
            ceps (ndarray, axis=(qerf,time)):
                output cepstrum (both sided, real). 
    """
    spec_db = np.log(spec)
    if dropphase:
        spec_db = spec_db.real + 0j
    ceps = np.fft.irfft(spec_db,axis=-2)
    return ceps.real


def ceps2spec(ceps):
    """
        Get spectrum from cepstrum.
        # Args
            ceps (ndarray, axis=(qerf,time)):
                input cepstrum (both sided, real).
        # Returns
            spec (ndarray, axis=(freq,time)):
                input specturm (one sided, complex). 
    """
    spec_db = np.fft.rfft(ceps,axis=-2)
    spec = np.exp(spec_db)
    return spec