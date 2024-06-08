# Reference

The examples referenced from https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention


# Difference Between Multi-Head Attention and Self-Attention

Multi-head attention and self-attention are key components in modern deep learning models, particularly in the Transformer architecture used for natural language processing (NLP). Here’s a breakdown of each and their differences:

## Self-Attention

**Definition:**
Self-attention, also known as intra-attention, is a mechanism that relates different positions of a single sequence to compute a representation of the sequence. It allows the model to focus on different parts of the input sequence when encoding or decoding.

**How It Works:**

1. **Input Vectors:** The input is a sequence of vectors (e.g., word embeddings).
2. **Query, Key, and Value:** For each input vector, three vectors are derived: a Query vector (Q), a Key vector (K), and a Value vector (V).
3. **Attention Scores:** The attention score is calculated using the dot product of the Query vector with all Key vectors. This gives a score representing the relevance of each word to the current word.
4. **Softmax:** The scores are normalized using the softmax function to get the attention weights.
5. **Weighted Sum:** The output is the weighted sum of the Value vectors, where the weights are the attention weights from the softmax.

Mathematically:
\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

Where \(d_k\) is the dimension of the Key vectors.

## Multi-Head Attention

**Definition:**
Multi-head attention involves running multiple self-attention mechanisms in parallel. Instead of performing a single attention function, multi-head attention projects the queries, keys, and values multiple times with different learned linear projections.

**How It Works:**

1. **Linear Projections:** The input vectors are linearly projected into \(h\) different sets of Q, K, and V vectors (where \(h\) is the number of heads).
2. **Parallel Attention Heads:** Each set of Q, K, and V vectors goes through the self-attention mechanism independently.
3. **Concatenation:** The outputs of these parallel attention heads are concatenated.
4. **Final Linear Projection:** The concatenated output is passed through a final linear layer to produce the final output.

Mathematically:
\[ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O \]
\[ \text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \]

Here, \(W_i^Q\), \(W_i^K\), \(W_i^V\) are the learned projection matrices for the \(i\)-th head, and \(W^O\) is the learned projection matrix for the final linear transformation.

## Key Differences

1. **Parallel Attention Mechanisms:**

   - **Self-Attention:** Uses a single attention mechanism to compute the attention scores and output.
   - **Multi-Head Attention:** Uses multiple parallel self-attention mechanisms (heads) to compute different sets of attention scores and outputs, which are then concatenated and transformed.
2. **Capacity to Focus on Different Aspects:**

   - **Self-Attention:** Can focus on different parts of the input sequence but is limited by a single set of projections.
   - **Multi-Head Attention:** Can focus on different parts of the input sequence simultaneously, each head potentially capturing different aspects of the input due to different learned projections.
3. **Representation Power:**

   - **Self-Attention:** Limited by a single attention mechanism.
   - **Multi-Head Attention:** Enhances the model's ability to capture more complex relationships and dependencies within the input sequence by combining multiple attention heads.

In summary, self-attention is a mechanism to compute the importance of different words within a sequence, while multi-head attention enhances this mechanism by running several self-attention layers in parallel to capture a richer representation of the input.
