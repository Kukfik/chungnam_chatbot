import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx, 1:3]).int()
        label = torch.tensor(self.label.iloc[idx, 3]).int()
        return sample, label
    
tensor_dataset = MyDataset('pytorch/test.csv')
dataset = dataloader(tensor_dataset, batch_size=4,shuffle = True)

for i, data in enumerate(dataset, 0):
    print(i)