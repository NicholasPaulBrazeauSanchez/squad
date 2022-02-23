"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
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
    
class gatedRNNEncoder1(nn.Module):
    """Encoder for the Microsoft model
    
    Output correspond to passage tokens, encoded with attention from the 
    question layer

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
        super(gatedRNNEncoder1, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=drop_prob if num_layers > 1 else 0.)
        self.gWeightEin = nn.Parameter(torch.zeros( hidden_size, hidden_size))
        self.gWeightZwei = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        for weight in (self.gWeightEin, self.gWeightZwei):
            nn.init.xavier_uniform_(weight)
            
        self.questProj = nn.Linear(hidden_size, 1, bias= False)
        self.passProj = nn.Linear(hidden_size, 1, bias = False)
        self.prevProj = nn.Linear(hidden_size, 1, bias = False)

    def forward(self, c, q, c_mask, q_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        softie = nn.Softmax(dim = 1)
        tanH = nn.Tanh()
        V = []
        vJ = torch.zeros_like(c[:, 0, :]).unsqueeze(1)
        for i in range (c_mask.shape[1]):
            sJ = self.questProj(q) + self.passProj(c[:, i, :]).unsqueeze(1) + self.prevProj(vJ) #(batch_size, q_len, 1)
            aT = softie(tanH(sJ.squeeze(2))) #(batch_size, 2 * hidden)
            cT = torch.bmm(aT.unsqueeze(1), q).squeeze(1).unsqueeze(0) #(1, batch_size, 2*hidden_size)
            # there's no need to put in gate layers here; the LSTM handles it for us
            vJ, _ = self.rnn(vJ, (c[:,i,:].unsqueeze(0), cT))
            vJ = F.dropout(vJ, self.drop_prob, self.training)
            V.append(vJ)
        return torch.stack(V, dim = 1).squeeze(2)
    
class selfAttentionRNNEncoder(nn.Module):
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
                 num_layers,
                 drop_prob=0.):
        super(selfAttentionRNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=drop_prob if num_layers > 1 else 0.)
        
        self.betterRnn = RNNEncoder(2 * hidden_size, hidden_size, num_layers = 1, drop_prob = drop_prob)
        self.gWeightEin = nn.Parameter(torch.zeros( hidden_size, hidden_size))
        self.gWeightZwei = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        
        self.selfAttnBest = nn.MultiheadAttention(hidden_size, num_heads = 1, 
                                              dropout = drop_prob, 
                                              batch_first= True)
        for weight in (self.gWeightEin, self.gWeightZwei):
            nn.init.xavier_uniform_(weight)
            
        self.currProj = nn.Linear(hidden_size, 1, bias = False)
        self.prevProj = nn.Linear(hidden_size, 1, bias = False)
        

    def forward(self, v, c_mask):
        # Save original padded length for use by pad_packed_sequence
        # may need to run v_0 through things
        
        '''
        nuevo = self.att_1(v)
        nuevoP = nuevo.unsqueeze(2).repeat(1, 1, v.shape[1], 1)
        print(nuevoP.shape)
        nuevoDos = self.att_2(v)
        nuevoDosP = nuevoDos.unsqueeze(2).repeat_interleave(1,1, v.shape[1], 1)
        print(nuevoDos.shape)
        confusion = tanH(nuevoP + nuevoDosP)
        a = masked_softmax(confusion, c_mask, dim=2)   
        print(a.shape)
        print(v.shape)
        '''
        
        #junk implementation
        '''
        H = []
        #make sure it's on the device
        for i in range (v.shape[1]):
            un  = self.currProj(v)
            deux = self.prevProj(v[:, i, :])
            sJ = tanH(un + deux.unsqueeze(1)).squeeze(2) #(batch_size, p_len, 1)
            #aT = softie(tanH(sJ.squeeze(2))) #(batch_size, 2 * hidden)
            aT = masked_softmax(sJ, c_mask, log_softmax=False)
            cT = torch.bmm(aT.unsqueeze(1), v) #(batch_size, 1, 2*hidden_size)

            base = torch.cat([v[:, i, :].unsqueeze(1), cT], dim = 2)
            H.append(base)
        Work = torch.stack(H, dim = 1).squeeze(2)
        Finale = self.betterRnn(Work, c_mask.sum(-1))
        return Finale
        '''
        
        key_mask = ~c_mask
        attended, _ = self.selfAttnBest(v, v, v, key_padding_mask = key_mask)
        #attended may not be enough?
        nuevo = torch.cat([v, attended], dim=2) 
        #yes! this works more effectively. Let's see about running this through
        #an RNN, though
        nuevoDos = self.betterRnn(nuevo, c_mask.sum(-1))
        
       # attended, _ = self.selfAttnBest(v, v, v)
        return nuevoDos
        
    
    
class DAFAttention(nn.Module):
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
        super(DAFAttention, self).__init__()
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

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)

        #x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        x = c * a
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


class SelfAttentionOutput(nn.Module):
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
        super(SelfAttentionOutput, self).__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1)
        self.rnn_linear_1 = nn.Linear(2 * hidden_size, 1)
        
        
        self.question_att = nn.Linear(2 * hidden_size, 1)
        self.ansPoint = torch.nn.RNN(input_size = 2 * hidden_size, 
                            hidden_size = 2 * hidden_size, 
                            num_layers = 1, batch_first = True)
        self.att_pos = nn.Linear(hidden_size, 1)
        
        #self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)
        self.rnn_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, q, q_mask, mask):
        # Shapes: (batch_size, seq_len, 1)
        # this is the default forward for this layer
        '''
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        '''
        #this is the fancy, rnn based forward for this layer. 
        
        # here, we may need additional input
        tanH = nn.Tanh()
        
        
        questAtt = tanH(self.question_att(q).squeeze(2)) # Shape: (batch, q_len)
        nu = masked_softmax(questAtt, q_mask, log_softmax= True)
        init = torch.bmm(nu.unsqueeze(1), q)
        
        s1 = tanH(self.att_linear_1(att) + self.rnn_linear_1(init))
        log_p1 = masked_softmax(s1.squeeze(), mask, log_softmax=True)
        a1 = masked_softmax(s1.squeeze(), mask, log_softmax=False)
        
        c1 = torch.bmm(a1.unsqueeze(1), att).squeeze(1).unsqueeze(0)
        
        h1, _ = self.ansPoint(init, c1)
        s1 = tanH(self.att_linear_2(att) + self.rnn_linear_2(h1))
        log_p2 = masked_softmax(s1.squeeze(), mask, log_softmax=True)
        

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
        #self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.att_linear_1 = nn.Linear(hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        
        self.question_att = nn.Linear(hidden_size, 1)
        self.ansPoint = torch.nn.RNN(input_size = 8 * hidden_size, 
                            hidden_size = hidden_size, 
                            num_layers = 1, batch_first = True)
        self.att_pos = nn.Linear(hidden_size, 1)
        
        #self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.att_linear_2 = nn.Linear(hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, q, q_mask, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        # this is the default forward for this layer
        '''
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        '''
        #this is the fancy, rnn based forward for this layer. 
        
        # here, we may need additional input
        questAtt = self.question_att(q).squeeze(2) # Shape: (batch, q_len)
        nu = masked_softmax(questAtt, q_mask, log_softmax= True)
        init = torch.bmm(nu.unsqueeze(1), q).squeeze(1).unsqueeze(0)
        
        att_1, nuStart  = self.ansPoint(att, init)
        
        att_1, nuStart  = self.ansPoint(att)
        logits_1 = self.att_linear_1(att_1) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        
        att_2, _ = self.ansPoint(att, nuStart)
        logits_2 = self.att_linear_2(att_2) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        
        

        return log_p1, log_p2
    
    
class BiDAFOutputGeneral(nn.Module):
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
        super(BiDAFOutputGeneral, self).__init__()
       # self.att_linear_1 = nn.Linear(hidden_size, 1)
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        
        self.question_att = nn.Linear(hidden_size, 1)
        self.ansPoint = torch.nn.RNN(input_size = 2 * hidden_size, 
                            hidden_size = hidden_size, 
                            num_layers = 1, batch_first = True)
        self.att_pos = nn.Linear(hidden_size, 1)
        
        #self.att_linear_2 = nn.Linear(hidden_size, 1)
        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)

    def forward(self, att, q, q_mask, mask):
        logits_1 = self.att_linear_1(att) 
        
        logits_2 = self.att_linear_2(att)
        
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        #this is the fancy, rnn based forward for this layer. 
        
        
        '''
        # It works fine, but I suspect I may not be using the attention correctly
        questAtt = self.question_att(q).squeeze(2) # Shape: (batch, q_len)
        nu = masked_softmax(questAtt, q_mask, log_softmax= True)
        init = torch.bmm(nu.unsqueeze(1), q).squeeze(1).unsqueeze(0)
        
        att_1, nuStart  = self.ansPoint(att, init)
        
        att_1, nuStart  = self.ansPoint(att)
        logits_1 = self.att_linear_1(att_1) 
        
        att_2, _ = self.ansPoint(att, nuStart)
        logits_2 = self.att_linear_2(att_2)
        
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        '''
        

        return log_p1, log_p2
