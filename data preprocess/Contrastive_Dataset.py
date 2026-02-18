import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, TensorData
from torch.utils.data import Dataset, DataLoader

class ContrastiveDataset(Dataset):
    def __init__(self,embeddings,X_positives,X_hard_negatives,labels,audio_nums,is_unlabelled):
        self.embeddings = embeddings
        self.X_positives = X_positives
        self.X_hard_negatives = X_hard_negatives
        self.labels = labels
        self.audio_nums = audio_nums
        self.is_unlabelled = is_unlabelled

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx],
                self.X_positives[idx],
                self.X_hard_negatives[idx],
                self.labels[idx],
                self.audio_nums[idx],
                self.is_unlabelled[idx])

# Create the dataset
contrastive_dataset = ContrastiveDataset(torch.tensor(embeddings).to(device),
                                         torch.tensor(X_positives).to(device),
                                         torch.tensor(X_hard_negatives).to(device),
                                         torch.tensor(labels).to(device),
                                         torch.tensor(audio_nums).to(device),
                                         torch.tensor(is_unlabelled).to(device))

train_loader = DataLoader(contrastive_dataset, batch_size=256, shuffle=False)
