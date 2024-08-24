import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionGen(nn.Module):
  def __init__(self,k,heads):
      super().__init__()
      assert k%heads==0
      self.k,self.heads=k,heads
      #The linear function creates a transformation layer which returns a transformed output for an input. y = x(W.T)
      #these also act as one of the parameters of the model.
      self.toKeys = nn.Linear(k,k,bias=False) #wk
      self.toQueries = nn.Linear(k,k,bias=False) #wq
      self.toValues = nn.Linear(k,k,bias=False) #wv

      #to project the resultant chunks of each attention head
      #combination can be done without transformation, so this projection isn't necessary.
      self.unifyHeads = nn.Linear(k,k)

  def forward(self,x, padding):
      #the inputs x and padding would be a 3-d vector of form (batch_size,seq_len,in_features) because the sequence of input vectors come batch by batch.
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
      # - fold heads into the batch dimension=> needed to compute the dot product parallely for each sequence.
      keys = keys.transpose(1, 2).contiguous().view(b * h, t, headSize)
      queries = queries.transpose(1, 2).contiguous().view(b * h, t, headSize)
      values = values.transpose(1, 2).contiguous().view(b * h, t, headSize)
      # now the first dimension is size of the batch: each batch has sequences of mini-vectors(one vector for each head).
      # size of sequence is the block size.
      # for each sequence, we have corresponding outputs
      # how to concatenate those outputs once you have transformed the matrix? we un-transform it first before concatenating.

      #compute weights
      raw_weights = torch.bmm(queries,keys.transpose(1,2))

      if(padding!=None):
        padding = padding.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)
        padding = padding.expand(b, h, t, t).contiguous().view(b * h, t, t)
        #make the weights corresponding to padded tokens -inf
        attention_mask = (padding==0)
        raw_weights.masked_fill_(mask=attention_mask,value=float('-inf'))
        
      #raw_weights is of dimension: b*h, t, t

      #disable forward looking
      indices = torch.triu_indices(t, t, offset=1)
      raw_weights[:, indices[0], indices[1]] = float('-inf')  
      # this + calculation of output vectors corresponding to padding input seems a bit

      raw_weights /= headSize**(1/2)
      weights = F.softmax(raw_weights,dim=-1)

      #apply self-attention to the input vectors
      out = torch.bmm(weights, values).view(b,h,t,headSize)
      out = out.transpose(1,2).contiguous().view(b,t,h*headSize)
      #unifyHeads is not really necessary once we do h*headSize
      return self.unifyHeads(out)
    
class GTransformerBlock(nn.Module):
  def __init__(self,k,heads):
    super().__init__()

    self.attention=SelfAttentionGen(k,heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
        nn.Linear(k,4*k),
        nn.ReLU(),
        nn.Linear(4*k,k)
    )
    #but sequential's units will output a number, not a vector? yep and a vector is k numbers
  def forward(self,x,padding):
    attended = self.attention(x,padding)
    x=self.norm1(attended+x)

    fedForward = self.ff(x)
    return (self.norm2(fedForward+x))
#will you need to return padding too, because what one layer returns is passed to another as input?

class GTransformer(nn.Module):
  def __init__(self,k,heads,depth,seq_length,num_tokens):
    super().__init__()

    #layer for handling input
    self.num_tokens=num_tokens #size of vocabulary i.e. no of unique tokens that the transformer knows.
    self.token_emb = nn.Embedding(num_tokens,k) # map each token(integer) to a size k vector.
    self.pos_emb = nn.Embedding(seq_length,k) # map each position (0->seq_length-1) to a size k vector
    #these embedding layers will be initialized randomly, but trained with the input.

    #transformer blocks
    tblocks=[]
    for i in range(depth):
      tblocks.append(GTransformerBlock(k,heads))
    self.tblocks = tblocks

    #layer for handling output: project to an array of size num_classes
    self.toProbs = nn.Linear(k,num_tokens)

  def forward(self,x,padding=None):
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

    for i,tblock in enumerate(self.tblocks):
       x = tblock.forward(x,padding)

    #process the output before returning
    x = self.toProbs(x) # projects x to a shape (b,t,num_tokens)
    #not converting to softmax or log_softmax here as output will be handled differently for train/test and inference.
    return x 