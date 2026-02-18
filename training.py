import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

scaler = GradScaler()

def combined_training_loop_fast(
    model, train_loader, optimizer, device, num_epochs=10, T=0.07,
    sup_weight=0.35, unsup_weight=0.65
):
    """
    model          : your encoder model
    train_loader   : yields (embeddings, pos, neg, label, audio_num, is_unlabelled)
    optimizer      : optimizer
    device         : torch.device
    num_epochs     : number of epochs
    T              : temperature for contrastive loss
    sup_weight     : weight for supervised loss
    unsup_weight   : weight for unsupervised loss
    """
    model.train()
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        epoch_sup_loss = 0.0
        epoch_unsup_loss = 0.0
        epoch_total_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for embeddings, pos, neg, label, audio_num, is_unlabelled in batch_pbar:
            # Move all tensors to device
            embeddings = embeddings.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B = embeddings.shape[0]
            P = pos.shape[1]
            K = neg.shape[1]

            # Flatten positives & negatives
            pos_flat = pos.reshape(B * P, *pos.shape[2:])
            neg_flat = neg.reshape(B * K, *neg.shape[2:])

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # --- Forward pass ---
                z = F.normalize(model(embeddings).squeeze(1), dim=1)         # [B, D]
                Z_pos_flat = F.normalize(model(pos_flat).squeeze(1), dim=1)  # [B*P, D]
                Z_neg_flat = F.normalize(model(neg_flat).squeeze(1), dim=1)  # [B*K, D]

                # Reshape back
                Z_pos = Z_pos_flat.reshape(B, P, -1)  # [B, P, D]
                Z_neg = Z_neg_flat.reshape(B, K, -1)  # [B, K, D]

                # --- Supervised loss ---
                labeled_mask = (label != -1)
                if labeled_mask.sum() > 1:
                    z_labeled = z[labeled_mask]
                    labels_labeled = label[labeled_mask]
                    sup_loss = supervised_contrastive_loss_vectorized(z_labeled, labels_labeled, T=T)
                else:
                    sup_loss = torch.tensor(0.0, device=device)

                # --- Unsupervised loss ---
                unsup_loss = unsupervised_contrastive_loss_vectorized(z, Z_pos, Z_neg, T=T)

                # --- Combined loss ---
                total_loss = sup_weight * sup_loss + unsup_weight * unsup_loss

            # --- Backward with AMP ---
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # optional: clip gradients if needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- Logging ---
            epoch_sup_loss += sup_loss.item()
            epoch_unsup_loss += unsup_loss.item()
            epoch_total_loss += total_loss.item()

            batch_pbar.set_postfix({
                "sup": f"{sup_loss.item():.4f}",
                "unsup": f"{unsup_loss.item():.4f}",
                "total": f"{total_loss.item():.4f}"
            })

        # --- Epoch summary ---
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"- SUP: {epoch_sup_loss:.4f}, "
            f"UNSUP: {epoch_unsup_loss:.4f}, "
            f"TOTAL: {epoch_total_loss:.4f}"
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
combined_training_loop_fast(model, train_loader, optimizer, device, num_epochs=50)
