from selfAttention import *

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    def forward(self, x_1, x_2):           # x_2 is new
        queries_1 = x_1 @ self.W_query
        
        keys_2 = x_2 @ self.W_key          # new
        values_2 = x_2 @ self.W_value      # new
        
        attn_scores = queries_1 @ keys_2.T # new 
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2
        return context_vec    

"""
Similar to SelfAttention, each context vector is a weighted sum of the values. 
However, in CrossAttention, these values are derived from the second input (x_2), 
and the weights are based on the interaction between x_1 and x_2.
"""

if __name__ == "__main__":
    torch.manual_seed(123)
    sentence = 'Life is short, eat dessert first'
    #tokenize it
    tokens = tokenize(sentence) 
    
    d_in, d_out_kq, d_out_v = 5, 2, 1
    embedded_sentence = embedding(tokens, d_in)
    d_in, d_out_kq, d_out_v = 3, 2, 4

    crossattn = CrossAttention(d_in, d_out_kq, d_out_v)

    first_input = embedded_sentence
    second_input = torch.rand(8, d_in)

    print("First input shape:", first_input.shape)
    print("Second input shape:", second_input.shape)

    context_vectors = crossattn(first_input, second_input)
    print(context_vectors)
    print("Output shape:", context_vectors.shape)

