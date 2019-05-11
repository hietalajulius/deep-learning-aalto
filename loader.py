import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval

device = torch.device('cpu')

path_to_processed = "data/dummy2.csv" 

class CustomDataset(Dataset):
    def __init__(self,path,train=True,mini=False):
        self.data = pd.read_csv(path)
        if (mini):
            self.data = self.data[:1000]
             
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = tensorFromIndices(self.data.indices[idx])
        y = self.data.target[idx]
        return X,y
    

def tensorFromIndices(indices):
    split = literal_eval(indices)
    return torch.tensor(split, dtype=torch.long).view(-1, 1)


def customCollate(list_of_samples):

    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    input_seqs,targets = list(zip(*list_of_samples))
    input_seq_lengths = [len(seq) for seq in input_seqs]
    targets = torch.tensor([tar for tar in targets])
    target_lengths = [1 for tar in targets]
    padding_value = 0
    
    pad_input_seqs = pad_sequence(input_seqs, batch_first=False, padding_value=padding_value)

    return pad_input_seqs, input_seq_lengths, targets, target_lengths

 
def getLoader():
    trainset = CustomDataset(path_to_processed,mini=True)
    loader = DataLoader(dataset=trainset,
                         batch_size=4,
                         shuffle=True,
                         collate_fn=customCollate,
                         pin_memory=True)
    return loader
