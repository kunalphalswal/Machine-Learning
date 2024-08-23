import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        self.toKeys = nn.Linear(k, k, bias=False) 
        self.toQueries = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)
        self.unifyHeads = nn.Linear(k, k)

    def forward(self, x, padding):
        b, t, k = x.size()
        h = self.heads
        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        headSize = self.k // self.heads
        keys = keys.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)
        queries = queries.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)
        values = values.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)

        raw_weights = torch.bmm(queries, keys.transpose(1, 2))
        padding = padding.unsqueeze(1).unsqueeze(2)
        padding = padding.expand(b, h, t, t).contiguous().view(b * h, t, t)

        attention_mask = (padding == 0)
        raw_weights.masked_fill_(attention_mask, float('-inf'))

        raw_weights /= headSize ** (1/2)
        weights = F.softmax(raw_weights, dim=-1)

        out = torch.bmm(weights, values).view(b, h, t, headSize)
        out = out.transpose(1, 2).contiguous().view(b, t, h * headSize)
        return self.unifyHeads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x, padding):
        attended = self.attention(x, padding)
        x = self.norm1(attended + x)
        fedForward = self.ff(x)
        return self.norm2(fedForward + x)

class CTransformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens 
        self.token_emb = nn.Embedding(num_tokens, k).to(self.device)
        self.pos_emb = nn.Embedding(seq_length, k).to(self.device)

        self.tblocks = nn.ModuleList([TransformerBlock(k, heads) for _ in range(depth)])

        self.toProbs = nn.Linear(k, num_classes).to(self.device)

    def forward(self, x, padding):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = torch.arange(t, device=self.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        x = tokens + positions

        for tblock in self.tblocks:
            x = tblock(x, padding)

        x = x.mean(dim=1)
        x = self.toProbs(x)
        return F.log_softmax(x, dim=1)
