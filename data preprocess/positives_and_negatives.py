import numpy as np
import torch
import torch.nn.functional as F

def positives(audio_nums, iterator, list_of_vectors, labels, num_positives=5):
   
    test_audio_num = audio_nums[iterator]

    if labels[iterator] != -1:
        same_audio_idx = np.where(
            (audio_nums == test_audio_num) & (labels != -1)
        )[0]

    # remove self
    same_audio_idx = same_audio_idx[same_audio_idx != iterator]

    if len(same_audio_idx) == 0:
        return np.empty((0, list_of_vectors.shape[1]))  # no positives

    # compute cosine similarity with anchor
    anchor = torch.tensor(list_of_vectors[iterator], dtype=torch.float32)
    candidates = torch.tensor(list_of_vectors[same_audio_idx], dtype=torch.float32)

    anchor_norm = F.normalize(anchor, dim=0)
    candidates_norm = F.normalize(candidates, dim=1)

    cos_sim = torch.matmul(candidates_norm, anchor_norm)  # [num_candidates]

    # select 'num_positives' least similar (smallest cosine similarity)
    num_to_select = min(num_positives, len(same_audio_idx))
    _, topk_idx = torch.topk(cos_sim, k=num_to_select, largest=False)  # least similar

    selected_indices = same_audio_idx[topk_idx.numpy()]

    positives = list_of_vectors[selected_indices]
    return positives

# Create X_positives
X_positives = []
for i in range(len(embeddings)):
    X_positives.append(hard_positives(audio_nums, i, embeddings, labels, num_positives=5))


def negatives(test_vector, list_of_vectors,num_negatives):
    test_vector = test_vector.reshape(1, -1)
    similarities = cosine_similarity(test_vector, list_of_vectors)[0]
    indices = np.argsort(similarities)[:num_negatives]
    neg_vectors = list_of_vectors[indices]

    return neg_vectors

num_negatives = 25
X_hard_negatives = []
for i in range(len(embeddings)):
    test = embeddings[i]
    X_hard_negatives.append(negatives(test, embeddings,num_negatives))
