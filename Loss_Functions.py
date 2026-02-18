import torch
import torch.nn.functional as F

EPS = 1e-12

def unsupervised_contrastive_loss_vectorized(z, Z_pos, Z_neg, T=0.1):
    """
    z: [B, D]
    Z_pos: [B, P, D]
    Z_neg: [B, K, D]
    returns scalar loss
    """
    # normalize
    z = F.normalize(z, dim=1)             # [B, D]
    Z_pos = F.normalize(Z_pos, dim=2)     # [B, P, D]
    Z_neg = F.normalize(Z_neg, dim=2)     # [B, K, D]

    # sim_pos: [B, P] = dot(Z_pos, z)
    # do batch matmul: (B, P, D) @ (B, D, 1) -> (B, P, 1)
    sim_pos = torch.bmm(Z_pos, z.unsqueeze(2)).squeeze(2) / T   # [B, P]

    # sim_neg: [B, K]
    sim_neg = torch.bmm(Z_neg, z.unsqueeze(2)).squeeze(2) / T   # [B, K]

    # numerators: exp(sim_pos)
    exp_pos = torch.exp(sim_pos)  # [B, P]

    # denominator per positive: exp_pos + sum(exp_neg) (broadcast)
    sum_exp_neg = torch.exp(sim_neg).sum(dim=1, keepdim=True)  # [B,1]
    denom = exp_pos + sum_exp_neg  # [B,P]

    loss_per_pos = - torch.log(exp_pos / (denom + EPS))  # [B,P]
    loss_per_anchor = loss_per_pos.mean(dim=1)          # [B]
    loss = loss_per_anchor.mean()
    return loss


def supervised_contrastive_loss_vectorized(z, labels, T=0.1):
    device = z.device
    L = z.size(0)
    if L <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Normalize embeddings
    z = F.normalize(z, dim=1)

    # 2. Compute similarity matrix and scale by temperature
    # sim[i, j] is the dot product (cosine sim) / T
    logits = torch.matmul(z, z.t()) / T

    # 3. For numerical stability: subtract max logit
    # This is the Log-Sum-Exp trick: exp(x - max) / sum(exp(x - max))
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # 4. Create masks
    mask_self = torch.eye(L, dtype=torch.bool, device=device)
    labels_col = labels.view(-1, 1)
    mask_pos = (labels_col == labels_col.t()) & ~mask_self

    # 5. Compute the denominator: sum(exp(logits)) for all j != i
    exp_logits = torch.exp(logits)
    # Mask out the self-similarity so it's not in the denominator
    exp_logits_masked = exp_logits.masked_fill(mask_self, 0.0)

    # denom_sum = sum_{j != i} exp(sim_ij)
    denom_sum = exp_logits_masked.sum(dim=1, keepdim=True)

    # 6. Compute Log-Probability
    # log(exp(P) / Sum) = P - log(Sum)
    # Add a tiny epsilon to denom_sum to prevent log(0) if no other samples exist
    log_prob = logits - torch.log(denom_sum + 1e-9)

    # 7. Mask and average over positives
    num_pos_per_anchor = mask_pos.sum(dim=1)

    # Only calculate loss for anchors that have at least one positive pair
    valid_anchors = num_pos_per_anchor > 0
    if not valid_anchors.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Sum log_probs of positives, then divide by count of positives
    pos_log_probs = (mask_pos.float() * log_prob).sum(dim=1)
    anchor_loss = -pos_log_probs[valid_anchors] / num_pos_per_anchor[valid_anchors]

    return anchor_loss.mean()

