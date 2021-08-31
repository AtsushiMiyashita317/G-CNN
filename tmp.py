import pickle

with open("phn.pickle",'rb') as f:
    x,y,z = pickle.load(f)
    print(x)
    print(y)
    print(z)