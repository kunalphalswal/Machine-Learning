import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,k,heads):
        super().__init__()
        assert k%heads==0
        self.k,self.heads=k,heads
        self.toKeys = nn.Linear(k,k,bias=False) 
        self.toQueries = nn.Linear(k,k,bias=False)
        self.toValues = nn.Linear(k,k,bias=False) 
        self.unifyHeads = nn.Linear(k,k)
    def forward(self,x):
        b,t,k=x.size()
        h=self.heads
        queries=self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        headSize = self.k//self.heads

        keys = keys.view(b,t,h,headSize)
        queries = queries.view(b,t,h,headSize)
        values = values.view(b,t,h,headSize)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, headSize)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, headSize)
        values = values.transpose(1, 2).contiguous().view(b * h, t, headSize)
        
        raw_weights = torch.bmm(queries,keys.transpose(1,2))
        
        raw_weights /= headSize**(1/2)
        weights = F.softmax(raw_weights,dim=2)

        out = torch.bmm(weights, values).view(b,h,t,headSize)
        out = out.transpose(1,2).view(b,t,h*headSize)
        return self.unifyHeads(out)

class TransformerBlock(nn.Module):
  def __init__(self,k,heads):
    super().__init__()

    self.attention=SelfAttention(k,heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
        nn.Linear(k,4*k),
        nn.ReLU(),
        nn.Linear(4*k,k)
    )
  def forward(self,x):
    attended = self.attention(x)
    x=self.norm1(attended+x)

    fedForward = self.ff(x)
    return self.norm2(fedForward+x)

class CTransformer(nn.Module):
  def __init__(self,k,heads,depth,seq_length,num_tokens,num_classes):
    super().__init__()

    self.num_tokens=num_tokens
    #needs an input of maximum seq_length elements and vocab_size as num_tokens
    self.token_emb = nn.Embedding(num_tokens,k) 
    self.pos_emb = nn.Embedding(seq_length,k)

    tblocks=[]
    for i in range(depth):
      tblocks.append(TransformerBlock(k,heads))
    self.tblocks=nn.Sequential(*tblocks)

    self.toProbs = nn.Linear(k,num_classes)

  def forward(self,x):
    tokens = self.token_emb(x)
    b, t, k = tokens.size()
    positions = torch.arange(t)
    positions = self.pos_emb(positions)[None,:,:].expand(b,t,k)

    x = tokens+positions

    x = self.tblocks(x)

    x = x.mean(dim=1) 
    x = self.toProbs(x) 
    return F.log_softmax(x,dim=1) 

