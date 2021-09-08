import argparse
import timit_data_processor
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="test class FramedTimit")
parser.add_argument("path", type=str, help="path to the directory that has annotation files")
args = parser.parse_args()

n_fft = 256

train_data = timit_data_processor.Timit(args.path,'train_annotations.csv','phn.pickle','data/',n_fft=n_fft)

train_dataloader = DataLoader(train_data, batch_size=128)
