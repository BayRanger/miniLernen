"""
https://spotintelligence.com/2023/01/31/self-attention/
In this example, the SelfAttention module takes an input tensor x with shape (batch_size, seq_length, input_dim)
and returns a weighted representation of the input sequence with the same form.

The attention mechanism is implemented using dot-product attention,
where the query, key, and value vectors are learned through linear transformations of the input sequence.

The attention scores are then calculated as the dot product of the queries and keys,
and the attention is applied by multiplying the values by the attention scores.

The result is a weighted representation of the input sequence that considers each element's importance.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim #also named embedding dimensiton
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

        #could it improve the performance?
        self.combine_qkv()


    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        qkv = self.qkv(x)

        q_new = qkv[...,:16]
        print(queries - q_new)

        print(f"value shape {list(values.shape)}")
        print(f"key shape {list(keys.shape)}")
        print(f"query shape {list(queries.shape)}")
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        print(f"weighed shape : {list(weighted.shape)}")
        return weighted

    def combine_qkv(self):
        def get_weight_bias(linear_layer):
            weights = linear_layer.weight.data  # Weight tensor
            bias = linear_layer.bias.data  # Bias tensor
            return weights,bias
        q_w,q_b = get_weight_bias(self.query)
        k_w,k_b = get_weight_bias(self.key)
        v_w,v_b = get_weight_bias(self.value)

        # this dimension trick is tricky!
        qkv_w  = torch.cat((q_w, k_w, v_w), dim=0)  # Concatenate along
        qkv_b  = torch.cat((q_b, k_b, v_b), dim=0)  # Concatenate along rows
        s_in, s_out = qkv_w.shape
        self.qkv = nn.Linear(s_in, s_out)
        self.qkv.weight.data = qkv_w
        self.qkv.bias.data = qkv_b

        #test_ln = nn.Linear(10,8)
        #print(test_ln.weight.data.shape)

        print(f"q_w,shape {q_w.shape}, q_b shape {q_b.shape}")
        #key_tensor = (self.key.weight.clone().detach())
        #value_tensor = torch.tensor(self.value.weight.clone().detach())


if __name__ == "__main__":
    batch_size = 1
    seq_length = 12
    input_dim = 16
    self_atten  = SelfAttention(input_dim)

    input_tensor = torch.rand(1, seq_length, input_dim)
    output_tensor = self_atten(input_tensor)
    torch.save(self_atten.state_dict(), 'atten_weights.pth')
