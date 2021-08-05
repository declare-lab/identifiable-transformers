''' Identifiable transformer '''
import torch

from torch import nn
from torch import Tensor
import torch.nn.functional as F

import multi_head_attention as M

from typing import Optional

'''
define model
'''
class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, concat_heads, dropout, bias=True, kdim=None, vdim=None):
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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()


    def forward(self, query, key, value, key_padding_mask, need_weights=False):
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
            key_padding_mask = key_padding_mask,
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
    def __init__(self, d_model, nhead, concat_heads, kdim, vdim, dim_feedforward, dropout):
        super().__init__(d_model, nhead)
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, concat_heads=concat_heads, dropout=dropout, kdim=kdim, vdim=vdim)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # requirements of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.activation = F.relu    #ReLU activation
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        #
    def forward(self, src, mask, return_attn_weights=False):
        src1, attn_weights = self.self_attn(query=src, key=src, value=src, key_padding_mask=mask, need_weights=return_attn_weights)
        src = src + self.dropout(src1)   #Currently all the dropouts happen with the same probability
        src = self.norm1(src)
        src1 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src1)
        src = self.norm2(src)
        return src, attn_weights


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, concat_heads, kdim, vdim, max_len, dim_feedforward, output_dim, 
                dropout, device, pos_emb, pad_id, return_attn_weights=False):
        super().__init__()

        #device cpu/gpu
        self.device = device

        #token embedding intialize pad_idx position with zeros and keep the gradient zero
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        #sentence encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, 
                                                     nhead=n_head, 
                                                     concat_heads=concat_heads, 
                                                     kdim=kdim, 
                                                     vdim=vdim, 
                                                     dim_feedforward=dim_feedforward,
                                                     dropout=dropout)
        
        #classification feed-forward layer
        self.fc = nn.Linear(embedding_dim, output_dim)

        #dropout regularisation
        self.dropout = nn.Dropout(dropout)

        #position embedding
        self.pos_emb = pos_emb
        if self.pos_emb == True:
            self.max_len = max_len
            self.positional_encoding = nn.Embedding(max_len, embedding_dim)

        #flag to return attention weights
        self.return_attn_weights = return_attn_weights

    def forward(self, mask, text):

        #[batch x max_len] --> [max_len x batch] (for MHA)
        text = text.transpose(0,1)

        #[max_len x batch] --> [max_len x batch]
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        #if positional embedding is required
        if self.pos_emb == True:
            pos_tensor = torch.tensor([[p for p in range(self.max_len)] for b in range(mask.shape[0])]).to(self.device)
            pos_encode = self.positional_encoding(pos_tensor)
            embedded = embedded + pos_encode.transpose(0,1).to(self.device)

        #[max_len x batch x embedding] --> [max_len x batch x embedding]
        hidden, attn_weights = self.encoder_layer(mask=mask, src=embedded, return_attn_weights=self.return_attn_weights)
        
        #feed-forward classification layer
        out = self.fc(hidden[0,:,:])
        
        #taking first token vector at output
        return out, attn_weights

