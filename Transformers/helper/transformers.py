import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,k,heads):
        super().__init__()
        assert k%heads==0
        self.k,self.heads=k,heads
        #The linear function creates a transformation layer which returns a transformed output for an input. y = x(W.T)
        self.toKeys = nn.Linear(k,k,bias=False) #wk
        self.toQueries = nn.Linear(k,k,bias=False) #wq
        self.toValues = nn.Linear(k,k,bias=False) #wv
        #to concatenate the resultant chunks of each attention head
        self.unifyHeads = nn.Linear(k,k)
    def forward(self,x):
        #the input would be a 3-d vector of form (batch_size,seq_len,in_features) because the sequence of input vectors comes in batches.
        b,t,k=x.size()
        h=self.heads
        #but how does the matrix multiplication work on 3d matrix such as x: it also gives out a 3d matrix of form (batch_size,seq_len,out_features)
        #here, both in and out features are k
        queries=self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        headSize = self.k//self.heads

        #This simply reshapes the tensors to break the last dimension into two dimensions.
        #purpose: to divide the features of each input vector into h parts, so that each head receives one chunk of that input vector.
        #the chunks are of low dimensions and easier to compute individually.
        keys = keys.view(b,t,h,headSize)
        queries = queries.view(b,t,h,headSize)
        values = values.view(b,t,h,headSize)

        # - fold heads into the batch dimension=> needed to compute the dot product parallely
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, headSize)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, headSize)
        values = values.transpose(1, 2).contiguous().view(b * h, t, headSize)
        # now the first dimension is size of the batch: each batch has sequences of mini-vectors(one vector for each head).
        # size of sequence is the block size.
        # for each sequence, we have corresponding outputs
        # how to concatenate those outputs once you have transformed the matrix? we un-transform it first before concatenating.

        #compute weights
        raw_weights = torch.bmm(queries,keys.transpose(1,2))
        #raw_weights is of dimension: b*h, t, t

        raw_weights /= headSize**(1/2)
        weights = F.softmax(raw_weights,dim=2)

        #apply self-attention to the input vectors
        out = torch.bmm(weights, values).view(b,h,t,headSize)
        out = out.transpose(1,2).contiguous().view(b,t,h*headSize)
        #unifyHeads is not really necessary once we do h*headSize
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
    #but sequential's units will output a number, not a vector? yep and a vector is k numbers
  def forward(self,x):
    attended = self.attention(x)
    x=self.norm1(attended+x)

    fedForward = self.ff(x)
    return self.norm2(fedForward+x)


class CTransformer(nn.Module):
  def __init__(self,k,heads,depth,seq_length,num_tokens,num_classes):
    super().__init__()

    #layer for handling input
    self.num_tokens=num_tokens #size of vocabulary i.e. no of unique tokens that the transformer knows.
    self.token_emb = nn.Embedding(num_tokens,k) # map each token(integer) to a size k vector.
    self.pos_emb = nn.Embedding(seq_length,k) # map each position (0->seq_length-1) to a size k vector
    #these embedding layers will be initialized randomly, but trained with the input.

    #transformer blocks
    tblocks=[]
    for i in range(depth):
      tblocks.append(TransformerBlock(k,heads))
    self.tblocks=nn.Sequential(*tblocks)

    #layer for handling output: project to an array of size num_classes
    self.toProbs = nn.Linear(k,num_classes)

  def forward(self,x):
    # process the input before feeding to transformer blocks
    """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
                  Each batch has t tokens. so each batch corresponds to 1 sequence?
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
                 Probability distribution over c classes for each batch
    """
    tokens = self.token_emb(x)
    b, t, k = tokens.size()
    positions = torch.arange(t,device=x.device)
    positions = self.pos_emb(positions)[None,:,:].expand(b,t,k)
    '''
    [None,:,:]
    This adds a new dimension at the beginning of the tensor.
    It changes the shape from (t, k) to (1, t, k).
    expand(b,t,k)
    This expands the tensor to shape (b, t, k), where b is the batch size.
    It repeats the positional embeddings b times along the first dimension.
    This operation doesn't allocate new memory; it creates a view of the original tensor.
    '''
    x = tokens+positions

    x = self.tblocks(x)

    #process the output before returning
    x = x.mean(dim=1) # calculates mean over the second dimension ie t. Now x is of shape (b,k) because each batch only has one vector
    x = self.toProbs(x) # projects x to a shape (b,1,num_classes)
    return F.log_softmax(x,dim=1) # calculates log of softmax across the second dimension ie num_classes. log is easier to handle than actual probability calculation
    #done according to the task of the transformer.

