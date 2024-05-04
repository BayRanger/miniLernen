"""
Reference: 
https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

"""



import torch

import torch.nn as nn

torch.manual_seed(123)



class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T  # unnormalized attention weights    
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec

def tokenize(sentence):
    #given a sentence
    dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
    sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    return sentence_int


def embedding(token, embed_dim = 3, vocab_size = 50_000):
    vocab_size = 50_000

    # it is just liner layer with idx search
    embed = torch.nn.Embedding(vocab_size, embed_dim)
    embedded_sentence = embed(token).detach()
    #print(f"embeded sentence size {list(embedded_sentence.shape)} ")
    # so we get emmbedding of words now
    return embedded_sentence


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T  # unnormalized attention weights    
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


def example_self_attention(tokens):
    # reduce d_out_v from 4 to 1, because we have 4 heads
    d_in, d_out_kq, d_out_v = 5, 2, 1
    embedded_sentence = embedding(tokens, d_in)
    
    
    #self attention with d_out_v heads
    sa = SelfAttention(d_in, d_out_kq, d_out_v)
    print(sa(embedded_sentence).shape)

def example_mutlihead_attention(tokens):

    d_in, d_out_kq, d_out_v = 5, 2, 1
    embedded_sentence = embedding(tokens, d_in)
     #multi head attention with num_heads
    #block_size = embedded_sentence.shape[1]
    #print(block_size)
    mha = MultiHeadAttentionWrapper(
        d_in, d_out_kq, d_out_v, num_heads=4
    )
    context_vecs = mha(embedded_sentence)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    
    sentence = 'Life is short, eat dessert first'
    #tokenize it

    tokens = tokenize(sentence) 
    #embedding it
    # in llama 2, the embedding size is 4096, here we use only 3

    """
    d: size of word vector x
    q:    [n, d_k]
    k:    [m, d_k]
    W_q:  [d, d_k]
    W_v:  [d, d_v]
    """
    #example_mutlihead_attention(tokens)
    example_self_attention(tokens)   
