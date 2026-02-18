import torch
from torch import nn, optim
import torch.nn.functional as F

class SDPA(nn.Module):
    def __init__(self, attn_do):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_do)

    def forward(self, query, key, value, mask):
        attn_scores = torch.einsum('bqnd,bknd->bnqk', [query, key])
        d_k = query.shape[-1]

        scaled_attn_scores = attn_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask != None:
            scaled_attn_scores.masked_fill_(mask==0, -torch.inf)

        attn_weights = F.softmax(scaled_attn_scores, dim=1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bnqk,bknd->bnqd', [attn_weights, value])

        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, attn_do=0, resid_do=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        head_dim = embed_dim // n_heads
        assert embed_dim == n_heads * head_dim, 'embed_dim should be perfectly divisible by n_heads'
        self.head_dim = head_dim

        self.proj_value, self.proj_key, self.proj_query = [nn.Linear(embed_dim, embed_dim) for _ in range(3)]

        self.sdpa = SDPA(attn_do)

        self.resid_dropout = nn.Dropout(resid_do)

        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, 'Embedding Dim of input tensor in forward is not equal to embed_dim passed in the contructor'

        value = self.proj_value(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        key = self.proj_key(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        query = self.proj_query(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        attn_output = self.sdpa(query, key, value, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.resid_dropout(self.final_proj(attn_output))

        return output

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, attn_do=0, resid_do=0, ff_hidden_mult=4, enc_do=0.0):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_heads, attn_do, resid_do)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim)
        )
        self.enc_dropout = nn.Dropout(enc_do)

    def forward(self, x, mask=None):
        attended = self.attention(x, mask)
        x = self.norm1(attended + x)
        x = self.enc_dropout(x)

        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)
        output = self.enc_dropout(x)

        return output


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_blocks):
        super().__init__()
        enc_blocks = []
        for i in range(n_blocks):
            enc_blocks.append(EncoderBlock(embed_dim, n_heads))

        self.enc = nn.Sequential(*enc_blocks)
        # self.logit_layer = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        output = self.enc(x.unsqueeze(1))

        return output

model = Encoder(embed_dim, n_heads, n_blocks)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
