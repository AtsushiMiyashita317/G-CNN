import pickle

with open("phn.pickle",'rb') as f:
    x = pickle.load(f)
    print(x)