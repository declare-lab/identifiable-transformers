''' Identifiable transformer '''
import torch

from torch import nn
from torch import Tensor
import torch.nn.functional as F

import multi_head_average as M

from typing import Optional

'''
define model
'''
class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, concat_heads, dropout=0., bias=True, kdim=None, vdim=None):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, kdim=kdim, vdim=vdim)

        #dimension of embedding vector
        self.embed_dim = embed_dim

        #dimension of key vector
        self.kdim = kdim

        #dimension of value vector, no longer = embed_dim // num_heads
        self.vdim = vdim

        #head dimension
        self.head_dim = vdim

        #no of heads in MHA
        self.num_heads = num_heads

        #dropout
        self.dropout = dropout

        #projection matrix to obtain query vector
        self.q_proj_weight = nn.Parameter(Tensor(embed_dim, self.kdim*num_heads))

        #projection matrix to obtain key vector
        self.k_proj_weight = nn.Parameter(Tensor(embed_dim, self.kdim*num_heads))

        #projection matrix to obtain value vector
        self.v_proj_weight = nn.Parameter(Tensor(embed_dim, self.vdim*num_heads))

        #initialize bias parameters for projection matrices
        self.in_proj_bias = nn.Parameter(torch.empty(kdim*num_heads*2 + vdim*num_heads))

        #head output: concatenate or add
        self.concat_head_output = concat_heads

        #weights for output transformation
        self.out_proj = nn.modules.linear._LinearWithBias(embed_dim, embed_dim)

        self._reset_parameters()


    def forward(self, query, key, value, text_len, need_weights=False):
        return M.multi_head_attention_forward(
            query=query, 
            key=key, 
            value=value,
            concat_head_output=self.concat_head_output,
            head_dim=self.head_dim,
            embed_dim_to_check=self.embed_dim, 
            num_heads=self.num_heads,
            in_proj_bias=self.in_proj_bias,
            dropout_p=self.dropout, 
            out_proj_weight=self.out_proj.weight, 
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            q_proj_weight=self.q_proj_weight, 
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight
            )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, concat_heads, kdim, vdim, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead)
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, concat_heads=concat_heads, kdim=kdim, vdim=vdim)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        #
    def forward(self, src: Tensor, text_len, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, text_len)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, concat_heads, kdim, vdim, max_len, dim_feedforward, output_dim, 
                dropout, device, pos_emb=False):
        super().__init__()
        self.device = device
        self.pos_emb = pos_emb
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #intialize pad_idx position with zeros and keep the gradient zero
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, concat_heads=concat_heads, kdim=kdim, vdim=vdim, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = nn.Parameter(torch.rand([max_len, embedding_dim], device=device)/max_len)


    def forward(self, text, text_lengths):

        #[batch x text_length] --> [text_length x batch] (for MHA)
        text = text.transpose(0,1)

        #[max_len x batch] --> [max_len x batch]
        embedded = self.dropout(self.embedding(text))

        #comment below if pos emb not required
        if self.pos_emb == True:
            mask = torch.zeros([text.shape[0],text.shape[1]], device=self.device)
            for sample in range(text.shape[1]):
                mask[:text_lengths[sample], sample] = torch.ones([text_lengths[sample]])
            embedded = embedded + self.positional_encoding[:text.shape[0],None,:]
            embedded = embedded * mask[:,:,None]

        #[encoder text_length x batch x embedding] --> [text_length x batch x embedding]
        hidden = self.encoder_layer(src = embedded, text_len = text_lengths)
        
        #feed-forward classification layer
        out = F.softmax(self.fc(hidden[0,:,:]), dim=-1)
        
        #taking first token vector at output
        return out

