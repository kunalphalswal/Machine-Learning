import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        # The linear function creates a transformation layer that returns a transformed output for an input. y = x(W.T)
        # These also act as one of the parameters of the model.
        self.toKeys = nn.Linear(k, k, bias=False)  # wk
        self.toQueries = nn.Linear(k, k, bias=False)  # wq
        self.toValues = nn.Linear(k, k, bias=False)  # wv

        # To project the resultant chunks of each attention head
        # Combination can be done without transformation, so this projection isn't necessary.
        self.unifyHeads = nn.Linear(k, k)

    def forward(self, x, padding):
        # The inputs x and padding would be a 3D vector of form (batch_size, seq_len, in_features) because the sequence of input vectors comes batch by batch.
        b, t, k = x.size()
        h = self.heads

        # Matrix multiplication works on a 3D matrix such as x and gives out a 3D matrix of form (batch_size, seq_len, out_features)
        # Here, both in and out features are k
        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        headSize = self.k // self.heads

        # This simply reshapes the tensors to break the last dimension into two dimensions.
        # Purpose: to divide the features of each input vector into h parts, so that each head receives one chunk of that input vector.
        # The chunks are of low dimensions and easier to compute individually.
        keys = keys.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)
        queries = queries.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)
        values = values.view(b, t, h, headSize).transpose(1, 2).contiguous().view(b * h, t, headSize)
        # Now the first dimension is the size of the batch: each batch has sequences of mini-vectors (one vector for each head).
        # Size of the sequence is the block size.
        # For each sequence, we have corresponding outputs.
        # How to concatenate those outputs once you have transformed the matrix? We un-transform it first before concatenating.

        # Compute weights
        raw_weights = torch.bmm(queries, keys.transpose(1, 2))
        padding = padding.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)
        padding = padding.expand(b, h, t, t).contiguous().view(b * h, t, t)

        # Make the weights corresponding to padded tokens -inf
        attention_mask = (padding == 0)
        raw_weights.masked_fill_(attention_mask, float('-inf'))
        # raw_weights is of dimension: b*h, t, t

        raw_weights /= headSize ** (1/2)
        weights = F.softmax(raw_weights, dim=-1)

        # Apply self-attention to the input vectors
        out = torch.bmm(weights, values).view(b, h, t, headSize)
        out = out.transpose(1, 2).contiguous().view(b, t, h * headSize)
        # unifyHeads is not really necessary once we do h*headSize
        return self.unifyHeads(out)

class CTransformerBlock(nn.Module):
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
        # But sequential's units will output a number, not a vector? Yep, and a vector is k numbers

    def forward(self, x, padding):
        attended = self.attention(x, padding)
        x = self.norm1(attended + x)

        fedForward = self.ff(x)
        return self.norm2(fedForward + x)
        # Will you need to return padding too, because what one layer returns is passed to another as input?

class CTransformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens  # Size of vocabulary i.e. number of unique tokens that the transformer knows.

        # Layer for handling input
        self.token_emb = nn.Embedding(num_tokens, k).to(self.device)  # Map each token (integer) to a size k vector.
        self.pos_emb = nn.Embedding(seq_length, k).to(self.device)  # Map each position (0->seq_length-1) to a size k vector
        # These embedding layers will be initialized randomly, but trained with the input.

        # Transformer blocks
        self.tblocks = nn.ModuleList([CTransformerBlock(k, heads) for _ in range(depth)])

        # Layer for handling output: project to an array of size num_classes
        self.toProbs = nn.Linear(k, num_classes).to(self.device)

    def forward(self, x, padding):
        # Process the input before feeding to transformer blocks
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
                  Each batch has t tokens. So each batch corresponds to 1 sequence?
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
                 Probability distribution over c classes for each batch
        """
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = torch.arange(t, device=self.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        '''
        [None,:,:]
        This adds a new dimension at the beginning of the tensor.
        It changes the shape from (t, k) to (1, t, k).
        expand(b,t,k)
        This expands the tensor to shape (b, t, k), where b is the batch size.
        It repeats the positional embeddings b times along the first dimension.
        This operation doesn't allocate new memory; it creates a view of the original tensor.
        '''
        x = tokens + positions

        for tblock in self.tblocks:
            x = tblock(x, padding)

        # Process the output before returning
        x = x.mean(dim=1)  # Calculates mean over the second dimension, i.e., t. Now x is of shape (b, k) because each batch only has one vector
        x = self.toProbs(x)  # Projects x to a shape (b, num_classes)
        return F.log_softmax(x, dim=1)  # Calculates log of softmax across the second dimension, i.e., num_classes. Log is easier to handle than actual probability calculation
        # Done according to the task of the transformer.
