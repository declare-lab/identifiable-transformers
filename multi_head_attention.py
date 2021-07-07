import torch

from torch import nn
from torch.nn.functional import linear, softmax, dropout

from functools import reduce
from typing import Optional, Tuple

import math

Tensor = torch.Tensor

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 head_dim: int,
                                 concat_head_output: bool,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_bias: Tensor,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 ):

    #query is [number of sentence tokens, batch, embedding dim]
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check

    #number of tokens and batch size should be same in key and value tensor
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    #scaling factor
    scaling = float(head_dim) ** -0.5

    #qh = qdim * num_heads
    _, qh = q_proj_weight.size()

    #qh = kdim * num_heads
    _, kh = k_proj_weight.size()

    #vh = vdim * num_heads
    _, vh = v_proj_weight.size()

    assert in_proj_bias.size()[0] == qh + kh + vh

    '''
    Input transformation
    : linear(x,w,b) = x*w^T + b
    '''

    #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, qh]
    q = linear(query, q_proj_weight.transpose(0, 1), in_proj_bias[0: qh])

    #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, kh]
    k = linear(key, k_proj_weight.transpose(0, 1), in_proj_bias[qh : qh + kh])

    #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, vh]
    v = linear(value, v_proj_weight.transpose(0, 1), in_proj_bias[qh + kh : ])
   
    #scaling query vectors
    q = q * scaling

    #[no of tokens, batch size * num_heads, qdim] -> [batch size * num_heads, no of tokens, qdim]
    q = q.view(tgt_len, bsz * num_heads, qh // num_heads).transpose(0, 1)

    #[no of tokens, batch size * num_heads, kdim] -> [batch size * num_heads, no of tokens, kdim]
    k = k.view(tgt_len, bsz * num_heads, kh // num_heads).transpose(0, 1)

    #[no of tokens, batch size * num_heads, vdim] -> [batch size * num_heads, no of tokens, vdim]
    v = v.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    #[batch size * num_heads, no of tokens, qdim] x [batch size * num_heads, no of tokens, kdim].T -> [batch size * num_head, no of tokens, no of tokens]
    attn_logits = torch.bmm(q, k.transpose(1, 2))

    #mask unwanted attentions from pad tokens
    if key_padding_mask != None:
        mask = key_padding_mask
        mask = mask.repeat(1,tgt_len)
        mask = mask.view(-1,tgt_len,tgt_len)
        mask = (mask*(mask.transpose(1,2))) == 1
        mask = mask.repeat(num_heads,1,1)
        attn_logits = attn_logits.masked_fill_(mask, -1e10)

    assert list(attn_logits.size()) == [bsz * num_heads, tgt_len, tgt_len]

    #softmax attention logits
    attn_output_weights = softmax(attn_logits, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    #[batch size * num_heads, no of tokens, no of tokens] * [batch size * num_heads, no of tokens, vdim].T  -> [batch size * num_heads, no of tokens, vdim]                                                   
    head_output = torch.bmm(attn_output_weights, v)

    assert list(head_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    if concat_head_output == True:
        #concat head outputs
        assert head_dim == embed_dim // num_heads
        head_output = head_output.transpose(0, 1).contiguous().view(tgt_len, bsz, num_heads * head_dim)

    elif concat_head_output == False:
        #add head outputs
        assert head_dim == embed_dim
        head_output = head_output.transpose(0, 1).contiguous().view(tgt_len, bsz, num_heads * head_dim)
        head_output = head_output.view(tgt_len, bsz, head_dim, num_heads)
        head_output = reduce(torch.add,[head_output[:,:,:,i] for i in range(head_output.size(3))])
    
    else:
        raise Exception("Unexpected type of operation over head outputs!")

    #output transformation
    head_output = linear(head_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, tgt_len)
        return head_output, attn_output_weights
    else:
        return head_output, None

