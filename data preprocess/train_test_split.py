import numpy as np
from collections import defaultdict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset

labelled = np.load('labelled_data_file') # Sl
print(labelled)
l_embeddings = labelled['test_embeddings']
l_labels = labelled['y_train']
l_class = labelled['y_labels']
l_audio_num = labelled['audio_num'] # for top POSITIVES as much as you want
labelled_ids = np.argmax(l_labels, axis=1)

unlabelled = np.load('unlablled_data_file') # Su

ul_embeddings = unlabelled['test_embeddings']
ul_labels = unlabelled['y_train']
ul_class = unlabelled["y_labels"]
ul_audio_num = unlabelled['audio_num']
unlabelled_ids = np.argmax(ul_labels, axis=1) +12 # so that label number don't match with labelled data labels

labelled_datax = {
    "embeddings": labelled["test_embeddings"],
    "y_labels": labelled_ids,
    "y_class" : l_class,
    "audio_num": labelled["audio_num"],
    #"transformations" : labelled['test_embeddings_shifted']
} # This is Sl

unlabelled_datax = {
    "embeddings": unlabelled["test_embeddings"],
    "y_labels": unlabelled_ids,    # will NOT use, as these are ground labels
     "y_class" : ul_class,
    "audio_num": unlabelled["audio_num"],
    #"transformations" : unlabelled['test_embeddings_shifted']
} # This is Su


def move_known_to_unlabelled(labelled_data, unlabelled_data, ratio=0.3, seed=42):
    np.random.seed(seed)

    # Unpack labelled data
    X = labelled_data["embeddings"]
    y_id = labelled_data["y_labels"]
    y_cls = np.array(labelled_data["y_class"])
    audio = labelled_data["audio_num"]

    # Containers
    keep_idx = []
    move_idx = []

    for cls in np.unique(y_cls):
        cls_idx = np.where(y_cls == cls)[0]
        np.random.shuffle(cls_idx)

        n_move = int(len(cls_idx) * ratio)
        move_idx.extend(cls_idx[:n_move])
        keep_idx.extend(cls_idx[n_move:])

    keep_idx = np.array(keep_idx)
    move_idx = np.array(move_idx)

    # --- New labelled (Sl) ---
    new_labelled = {
        "embeddings": X[keep_idx],
        "y_labels": y_id[keep_idx],
        "y_class": y_cls[keep_idx].tolist(),
        "audio_num": audio[keep_idx]
    }

    # --- Known samples moved to Su ---
    known_unlabelled = {
        "embeddings": X[move_idx],
        "y_labels": y_id[move_idx],     # keep ONLY for eval
        "y_class": y_cls[move_idx],
        "audio_num": audio[move_idx]
    }

    # --- Merge with existing unlabeled (novel classes) ---
    new_unlabelled = {
        "embeddings": np.concatenate([
            unlabelled_data["embeddings"],
            known_unlabelled["embeddings"]
        ]),
        "y_labels": np.concatenate([
            unlabelled_data["y_labels"],
            known_unlabelled["y_labels"]
        ]),
        "y_class": np.concatenate([
            unlabelled_data["y_class"],
            known_unlabelled["y_class"]
        ]),
        "audio_num": np.concatenate([
            unlabelled_data["audio_num"],
            known_unlabelled["audio_num"]
        ])
    }

    return new_labelled, new_unlabelled


labelled_data, unlabelled_data = move_known_to_unlabelled(
    labelled_datax,
    unlabelled_datax,
    ratio=0.3  # 30% known to  Su
)
# WE DID THIS AS OUR UNLABELLED DATA COMTAINED NOVEL CLASSES ONLY , SO WE ADDED SOME PERCENTAGE OF DATAPOINTS IN UNLABELLED DATA TO MATCH THE PROBLEM
