"""Assortment of layers for use in models.py.
Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb
    
class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(EmbeddingWithChar, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.embedChar = nn.Embedding.from_pretrained(char_vectors, freeze=False, padding_idx = 0)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.conv = nn.Conv2d(in_channels = char_vectors.size(1), 
                              out_channels = hidden_size,
                              kernel_size = (1,5), bias = True)
        #probably bad to hardcode shapes like this, but it'll do
        self.maxpool = nn.MaxPool2d((1,12))
        self.hwy = HighwayEncoder(2, 2 * hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, xchar, xmasks):

        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        #this is legit
        embChar = self.embedChar(xchar) # (batch_Size, seq_len, char_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        embChar = F.dropout(embChar, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        #hit embChar with a 2dconv, and then a highway
        embChar = embChar.permute((0, 3, 1, 2))
        embChar = self.conv(embChar)# (batch_size, seq_len, hidden_size)
        # I don't think the relu is helping
        #embChar = F.relu(embChar)
        
        
       # embChar = F.dropout(embChar, self.drop_prob, self.training)
        
        embChar = self.maxpool(embChar).squeeze(3)
        
        #embChar, _ = torch.max(embChar, dim = 3)
        embChar = torch.transpose(embChar, 1, 2)
       # embChar = F.dropout(embChar, self.drop_prob, self.training)
        proc = torch.cat([emb, embChar], dim = 2)
        proc = self.hwy(proc)   # (batch_size, seq_len, hidden_size)

        return proc


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, J??rgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class ForwardRNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(ForwardRNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x
    
    
class selfAttention(nn.Module):
    """Self attention RNN encoder
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_prob=0.):
        super(selfAttention, self).__init__()
        self.drop_prob = drop_prob
        
        self.Rnn = RNNEncoder(8 * hidden_size , 2* hidden_size, 1, drop_prob = 0)
        self.selfAttn = nn.MultiheadAttention(2 * hidden_size, num_heads = 1, 
                                              batch_first= True)
        self.RelevanceGate = nn.Linear(4 * hidden_size, 4 * hidden_size, bias = False)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        key_mask = ~c_mask
        attended, _ = self.selfAttn(v, v, v, key_padding_mask = key_mask)
        #attended may not be enough?
        nuevo = torch.cat([v, attended], dim=2) 
        gate = torch.sigmoid(self.RelevanceGate(nuevo))
        nuevo = gate * nuevo
        nuevoDos = self.Rnn(nuevo, c_mask.sum(-1))
        nuevoDos = F.dropout(nuevoDos, self.drop_prob, self.training)
        return nuevoDos

class selfAttention2(nn.Module):
    """Self attention RNN encoder
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_prob=0.):
        super(selfAttention2, self).__init__()
        self.drop_prob = drop_prob
        
        self.Rnn = RNNEncoder(2 * input_size , hidden_size, 1, drop_prob = drop_prob)
        self.selfAttn = nn.MultiheadAttention(input_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True, bias = False)
        self.RelevanceGate = nn.Linear(2 * input_size, 2 * input_size, bias = False)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        key_mask = ~c_mask
        attended, _ = self.selfAttn(v, v, v, key_padding_mask = key_mask)
        #attended may not be enough?
        nuevo = torch.cat([v, attended], dim=2) 
        gate = torch.sigmoid(self.RelevanceGate(nuevo))
        nuevo = gate * nuevo
        nuevoDos = self.Rnn(nuevo, c_mask.sum(-1))
        return nuevoDos
    
class selfAttention3(nn.Module):
    """Self attention RNN encoder
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_prob=0.):
        super(selfAttention3, self).__init__()
        self.drop_prob = drop_prob
        
        self.Rnn = RNNEncoder(2 * input_size , hidden_size, 1, drop_prob = drop_prob)
        self.Projector = nn.Linear(2 * input_size, 2 * hidden_size, bias = False)
        self.selfAttn = nn.MultiheadAttention(input_size, num_heads = 1, 
                                              batch_first= True, bias = False)
        self.RelevanceGate = nn.Linear(2 * input_size, 2 * input_size, bias = False)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        key_mask = ~c_mask
        attended, _ = self.selfAttn(v, v, v, key_padding_mask = key_mask)
        attended = F.dropout(attended, self.drop_prob, self.training)
        #attended may not be enough?
        nuevo = torch.cat([v, attended], dim=2) 
        #gate = torch.sigmoid(self.RelevanceGate(nuevo))
        #nuevoDos = gate * nuevo
        nuevoDos = self.Rnn(nuevo, c_mask.sum(-1))
        return nuevoDos
    
    
class CoAttention(nn.Module):
    """
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(CoAttention, self).__init__()
        self.drop_prob = drop_prob
        # self.biLSTM = nn.LSTM(hidden_size=hidden_size, bidirectional=True)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        # print("\nBatch_size. c_len, q_len", batch_size, c_len, q_len)
        # print("c size",  c.size())
        # print("q size", q.size())
        L = torch.bmm(c, q.transpose(1, 2))  # (batch_size, c_len, q_len)
        # print("L size", L.size())
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)

        ac = masked_softmax(L, q_mask, dim=2)  # (batch_size, c_len, q_len)
        aq = masked_softmax(L, c_mask, dim=1)  # (batch_size, c_len, q_len)
        # print("ac size", ac.size())
        # print("aq size", aq.size())

        Cq = torch.bmm(c.transpose(1,2), aq)
        # print("Cq size", Cq.size())
        # 64, 200, 23
        # 64, 23, 200
        concatenated = torch.cat((q.transpose(1,2),Cq), axis=1)
        # print("concatenated size", concatenated.size())
        Cc = torch.bmm(concatenated, ac.transpose(1,2))
        # print("CC size", Cc.size())

        out = torch.cat((c, Cc.transpose(1,2)), axis=2)
        # print("out size", out.size())
        return out
    


class encoderBlock(nn.Module):
    """Self attention RNN encoder
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_prob=0.):
        super(selfAttention, self).__init__()
        #add positional encoding! Use sin functions
        self.drop_prob = drop_prob
        
        self.Rnn = RNNEncoder(4 * hidden_size , hidden_size, 1, drop_prob = drop_prob)
        self.selfAttn = nn.MultiheadAttention(2 * hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True)
        #self.Convoluter = [ , nn.Linear(2 * hidden_size)]
        self.LinearEin = nn.Linear(2 * hidden_size)
        self.layerNormEin = nn.Linear(2 * hidden_size)
        self.LinearZwei = nn.Linear(2 * hidden_size)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        key_mask = ~c_mask
        v = v + self.positionEncoder(v)
        v = torch.relu(self.selfLinearZero(v))
        attended, _ = self.selfAttn(v, v, v, key_padding_mask = key_mask)
        #attended may not be enough?
        
        attended = F.relu(self.LinearEin(attended))
        attended = F.relu(self.LinearEin(attended))
        nuevo = torch.cat([v, attended], dim=2) 
        gate = torch.sigmoid(self.RelevanceGate(nuevo))
        gate = F.dropout(gate, self.drop_prob, self.training)
        nuevo = gate * nuevo
        nuevoDos = self.Rnn(nuevo, c_mask.sum(-1))
        return nuevoDos
    
    # https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    def positionEncoder(self, v):
        d = v.shape[2]
        encoder = nn.zeros_like(v[1,:,:])
        pos  = torch.arange(0, v.shape[1]).unsqueeze(1)
        divisor = torch.exp((torch.arange(0, d, 2, dtype=torch.float)* -(math.log(10000.0) / d)))
        encoder[:, 0::2] = torch.sin(pos.float() * divisor)
        encoder[:, 1::2] = torch.cos(pos.float() * divisor)
        return encoder
        
        
    
class selfAttentionBiDAF(nn.Module):
    """Self attention RNN encoder
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_prob=0.):
        super(selfAttentionBiDAF, self).__init__()
        self.drop_prob = drop_prob
        
        self.Rnn = RNNEncoder(4 * hidden_size , hidden_size, 1, drop_prob = drop_prob)
        self.selfAttn = nn.MultiheadAttention(2 * hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True)
        self.RelevanceGate = nn.Linear(4 * hidden_size, 4 * hidden_size, bias = False)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        key_mask = ~c_mask
        attended, _ = self.selfAttn(v, v, v, key_padding_mask = key_mask)
        #attended may not be enough?
        nuevo = torch.cat([v, attended], dim=2) 
        gate = torch.sigmoid(self.RelevanceGate(nuevo))
        #gate = F.dropout(gate, self.drop_prob, self.training)
        nuevo = gate * nuevo
        nuevo = F.dropout(nuevo, self.drop_prob, self.training)
        nuevoDos = self.Rnn(nuevo, c_mask.sum(-1))
        return nuevoDos
        
        
    
    
class DAFAttention(nn.Module):
    """As BiDAF attention, but we only keep track of context-to-question
    attention
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(DAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        self.RelevanceGate = nn.Linear(2 * hidden_size, 2 * hidden_size, bias = False)

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        preserved = c_mask.sum(-1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        incoming = torch.cat([c, a], dim = 2)

        #x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        gate = torch.sigmoid(self.RelevanceGate(incoming))
        incoming = gate * incoming
        #processed = self.matcher(incoming, preserved)
        #processed = F.dropout(processed, self.drop_prob, self.training)
        return incoming
    
    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class SelfAttentionRNNOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    Uses the question encodings to get an input state to an attention 
    Calculation involving our hidden states and the question encodings.
    This attention is then used with the question encodings to get a new 
    hidden state that we use to get another attention calculation, representing
    the start pointers and end pointers
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(SelfAttentionRNNOutput, self).__init__()
        self.attention_size = hidden_size
        self.att_linear_1 = nn.Linear(4 * hidden_size, self.attention_size, 
                                      bias = False)
        self.rnn_linear_1 = nn.Linear(2 * hidden_size, self.attention_size, 
                                      bias = False)
        
        
        
        self.question_att = nn.Linear(2 * hidden_size, self.attention_size)
        self.ansPoint = torch.nn.RNN(input_size = 4 * hidden_size, 
                            hidden_size = 2 * hidden_size, 
                            num_layers = 1, dropout = drop_prob, batch_first = True)
        
        # this RNN probably isn't applying context masking correctly, 
        # but I'm not sure how to fix it
        self.att_layer = nn.Linear(self.attention_size, 1, bias = False)
        self.tanH =  nn.Tanh()
        self.drop_prob = drop_prob
        
        # these layers were originally for predicting the end pointer
        # but they've been done away with
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)
        self.rnn_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, q, q_mask, mask):
        #this is the fancy, rnn based forward for this layer. 
        
        # here, we may need additional input
       
        
        questAtt = self.att_layer(self.tanH(self.question_att(q).squeeze(2))) # Shape: (batch, q_len, 1)
        nu = masked_softmax(questAtt.squeeze(2), q_mask, log_softmax= False)
        init = torch.bmm(nu.unsqueeze(1), q)
        
        att_linear = self.att_linear_1(att) #
        
        s1 = self.att_layer(self.tanH(att_linear + self.rnn_linear_1(init))) # (batch, c_len, 1)
        log_p1 = masked_softmax(s1.squeeze(), mask, log_softmax=True)
        a1 = masked_softmax(s1.squeeze(), mask, log_softmax=False) 
        c1 = torch.bmm(a1.unsqueeze(1), att) #(batch, c_len, 4 * hidden)
        
        h1, _ = self.ansPoint(c1, init.transpose(0,1))
        h1 = F.dropout(h1, self.drop_prob, self.training)
        s2 = self.att_layer(self.tanH(att_linear  + self.rnn_linear_1(h1))) # (batch, c_len,1)
        log_p2 = masked_softmax(s2.squeeze(), mask, log_softmax=True)
        

        return log_p1, log_p2
    
class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class BiDAFOutputRnn(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutputRnn, self).__init__()
        self.drop_prob = drop_prob
        self.attn_size = hidden_size
        self.lastState = nn.Linear(2 * hidden_size, self.attn_size)
        self.Attn1 = nn.Linear(8 * hidden_size, self.attn_size)
        self.Attn2 = nn.Linear(8 * hidden_size, self.attn_size)
        self.attn_proj = nn.Linear(self.attn_size, 1, bias = False)
        self.question_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        
       # self.ansPoint = ForwardRNNEncoder(2 * hidden_size, 2 * hidden_size, 1, 
                         #                 drop_prob = drop_prob)
        
        self.ansPoint = torch.nn.RNN(input_size = 2 * hidden_size, 
                            hidden_size = 2 * hidden_size, 
                            num_layers = 1, dropout = drop_prob, batch_first = True)
        
        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size= hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        
        self.selfAttn = nn.MultiheadAttention(2* hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True, bias = False)
        
        self.init_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.attnInit = nn.Parameter(torch.zeros(1, 2 * hidden_size))
        nn.init.xavier_uniform_(self.attnInit)
        
        self.modState = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.modState2 = nn.Linear(2 * hidden_size, self.attn_size, bias = False)


    def forward(self, att, q, q_mask, mod, mask):
        questAtt = self.attn_proj(torch.tanh(self.question_attn(q).squeeze(2) + self.init_attn(self.attnInit))) # Shape: (batch, q_len, 1)
        #questAtt = self.question_attn_var(q)
        nu = masked_softmax(questAtt.squeeze(2), q_mask, log_softmax= False)
        init = torch.bmm(nu.unsqueeze(1), q)
        '''
        repo = self.attnInit.repeat(q.shape[0], 1, 1)
        init, _ = self.selfAttn(repo, q, q, key_padding_mask = ~q_mask)
        '''
        
        logits_1 = self.attn_proj(self.Attn1(att) + self.modState(mod) + self.lastState(init))
        b1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=False)
        WeightedB1 = torch.bmm(b1.unsqueeze(1), mod)
        new, _ = self.ansPoint(WeightedB1, torch.transpose(init, 0, 1))
        new = F.dropout(new, self.drop_prob, self.training)
        mod = self.rnn(mod, mask.sum(-1))
        
        logits_2 = self.attn_proj(torch.tanh(self.Attn2(att) + self.modState2(mod) + self.lastState(new)))
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class BiDAFOutputRnnCoatt(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutputRnnCoatt, self).__init__()
        self.drop_prob = drop_prob
        self.attn_size = hidden_size
        self.lastState = nn.Linear(2 * hidden_size, self.attn_size)
        self.Attn1 = nn.Linear(6 * hidden_size, self.attn_size)
        self.Attn2 = nn.Linear(6 * hidden_size, self.attn_size)
        self.attn_proj = nn.Linear(self.attn_size, 1, bias = False)
        self.question_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        
       # self.ansPoint = ForwardRNNEncoder(2 * hidden_size, 2 * hidden_size, 1, 
                         #                 drop_prob = drop_prob)
        
        self.ansPoint = torch.nn.RNN(input_size = 2 * hidden_size, 
                            hidden_size = 2 * hidden_size, 
                            num_layers = 1, dropout = drop_prob, batch_first = True)
        
        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size= hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        
        self.selfAttn = nn.MultiheadAttention(2* hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True, bias = False)
        
        self.init_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.attnInit = nn.Parameter(torch.zeros(1, 2 * hidden_size))
        nn.init.xavier_uniform_(self.attnInit)
        
        self.modState = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.modState2 = nn.Linear(2 * hidden_size, self.attn_size, bias = False)


    def forward(self, att, q, q_mask, mod, mask):
        questAtt = self.attn_proj(torch.tanh(self.question_attn(q).squeeze(2) + self.init_attn(self.attnInit))) # Shape: (batch, q_len, 1)
        #questAtt = self.question_attn_var(q)
        nu = masked_softmax(questAtt.squeeze(2), q_mask, log_softmax= False)
        init = torch.bmm(nu.unsqueeze(1), q)
        '''
        repo = self.attnInit.repeat(q.shape[0], 1, 1)
        init, _ = self.selfAttn(repo, q, q, key_padding_mask = ~q_mask)
        '''
        
        logits_1 = self.attn_proj(self.Attn1(att) + self.modState(mod) + self.lastState(init))
        b1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=False)
        WeightedB1 = torch.bmm(b1.unsqueeze(1), mod)
        new, _ = self.ansPoint(WeightedB1, torch.transpose(init, 0, 1))
        new = F.dropout(new, self.drop_prob, self.training)
        mod = self.rnn(mod, mask.sum(-1))
        
        logits_2 = self.attn_proj(torch.tanh(self.Attn2(att) + self.modState2(mod) + self.lastState(new)))
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class BiDAFOutputRnnMulti(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutputRnnMulti, self).__init__()
        self.drop_prob = drop_prob
        self.attn_size = hidden_size
        self.lastState = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.lastStateVar = nn.Linear(2 * hidden_size, 1)
        self.Attn1 = nn.Linear(8 * hidden_size, self.attn_size, bias = False)
        self.Attn1var = nn.Linear(8 * hidden_size, 1)
        self.Attn2var = nn.Linear(8 * hidden_size, 1)
        self.Attn2 = nn.Linear(8 * hidden_size, self.attn_size, bias = False)
        self.attn_proj = nn.Linear(self.attn_size, 1, bias = False)
        self.question_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.question_attn_var = nn.Linear(2 * hidden_size, 1)
        
       # self.ansPoint = ForwardRNNEncoder(2 * hidden_size, 2 * hidden_size, 1, 
                         #                 drop_prob = drop_prob)
        
        self.ansPoint = torch.nn.RNN(input_size = 2 * hidden_size, 
                            hidden_size = 2 * hidden_size, 
                            num_layers = 1, dropout = drop_prob, batch_first = True)
        
        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size= hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        
        self.init_attn = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.selfAttn = nn.MultiheadAttention(2* hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True)
        
        self.attnInit = nn.Parameter(torch.zeros(1, 2 * hidden_size))
        self.initProj = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        nn.init.xavier_uniform_(self.attnInit)
        
        self.modState = nn.Linear(2 * hidden_size, self.attn_size, bias = False)
        self.modStateVar = nn.Linear(2 * hidden_size, 1, bias = False)
        self.modStateVar2 = nn.Linear(2 * hidden_size, 1, bias = False)
        self.modState2 = nn.Linear(2 * hidden_size, self.attn_size, bias = False)


    def forward(self, att, q, q_mask, mod, mask):
        
        questAtt = self.attn_proj(torch.tanh(self.question_attn(q).squeeze(2) + self.init_attn(self.attnInit))) # Shape: (batch, q_len, 1)
        #questAtt = self.question_attn_var(q)
        nu = masked_softmax(questAtt.squeeze(2), q_mask, log_softmax= False)
        init = torch.bmm(nu.unsqueeze(1), q)
        
        # this is the best yet!
        #repo = self.attnInit.repeat(q.shape[0], 1, 1)
        #init, _ = self.selfAttn(repo, q, q, key_padding_mask = ~q_mask)
        logits_1 = self.attn_proj(torch.tanh(self.modState(mod) + self.lastState(init)))
        #logits_1 = self.Attn1var(att) + self.modStateVar(mod) + self.lastStateVar(init)
        b1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=False)
        WeightedB1 = torch.bmm(b1.unsqueeze(1), mod)
        new, _ = self.ansPoint(WeightedB1, torch.transpose(init, 0, 1))
        new = F.dropout(new, self.drop_prob, self.training)
        #mod = self.rnn(mod, mask.sum(-1))
        
        logits_2 = self.attn_proj(torch.tanh(self.modState(mod) + self.lastState(new)))
        #logits_2 = self.Attn2var(att) + self.modStateVar2(mod) + self.lastStateVar(new)
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class SelfAttnOutputPtr(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(SelfAttnOutputPtr, self).__init__()
        self.drop_prob = drop_prob
        self.attn_size = 75
        self.lastState = nn.Linear(4 * hidden_size, self.attn_size)
        self.curAttn = nn.Linear(8 * hidden_size, self.attn_size)
        self.attn_proj = nn.Linear(self.attn_size, 1)
        
        self.ansPoint = ForwardRNNEncoder(2 * hidden_size, 2 * hidden_size, 1, 
                                          drop_prob = drop_prob)
        
        self.ansPoint = nn.RNN(8 * hidden_size, 4 * hidden_size)
        
        self.rnn = RNNEncoder(input_size=8 * hidden_size,
                              hidden_size= 2 * hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        
        self.modState = nn.Linear(4 * hidden_size, self.attn_size)

    def forward(self, att, q, q_mask, mod, mask):
        
        #maybe factor in attn, but put it over two linear layers
        
        
        logits_1 = self.attn_proj(torch.tanh(self.modState(mod) + self.lastState(torch.zeros_like(mod))))
        b1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=False)
        WeightedB1 = b1.unsqueeze(2) * mod
        new, _ = self.ansPoint(WeightedB1)
        new = F.dropout(new, self.drop_prob, self.training)
        
        logits_2 = self.attn_proj(torch.tanh(self.modState(mod) + self.lastState(new)))
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class LinearSelfAttentionOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(LinearSelfAttentionOutput, self).__init__()
        self.att_linear_1 = nn.Linear(6 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(6 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
