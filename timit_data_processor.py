import argparse

import numpy as np
import pickle
import pandas as pd
import soundfile as sf
from scipy import signal


phn_dict = {}
phn_list = []
phn_count = 0

def feature_extraction(path,fwfdst):
    """
    """
    df = pd.read_csv(path)
    
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

        sign, sr = sf.read(x_df_wav[i])
        spec = signal.stft(sign,sr,nperseg=n_fft)[2]

        df = pd.read_csv(x_df_phn[i],delimiter=' ',header=None)
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


def main():
    parser = argparse.ArgumentParser(description="process timit data")
    parser.add_argument("path", type=str, help="path to the directory that has data-path files")
    parser.add_argument("dst", type=str, help="feature is saved as the file")
    args = parser.parse_args()

    x_train,y_train = feature_extraction(f"{args.path}/train_data.csv",f"{args.path}/train_fwf.csv")
    x_test,y_test = feature_extraction(f"{args.path}/test_data.csv",f"{args.path}/test_fwf.csv")

    np.savez(args.dst,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

    f = open('phn.pickle', 'wb')
    pickle.dump(phn_list, f)
    
    


if __name__=="__main__":
    main()