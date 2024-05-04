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
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        print(f"value shape {list(values.shape)}")
        print(f"key shape {list(keys.shape)}")
        print(f"query shape {list(queries.shape)}")
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        print(f"weighed shape : {list(weighted.shape)}")
        return weighted
    
if __name__ == "__main__":
    batch_size = 1
    seq_length = 12
    input_dim = 16
    self_atten  = SelfAttention(input_dim)
    input_tensor = torch.rand(1, seq_length, input_dim)
    output_tensor = self_atten(input_tensor)
    torch.save(self_atten.state_dict(), 'atten_weights.pth')
 