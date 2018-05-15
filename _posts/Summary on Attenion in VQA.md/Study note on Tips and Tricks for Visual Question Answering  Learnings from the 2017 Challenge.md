---
title: Study note on Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge
---

*Damien Teney, Peter Anderson, Xiaodong He, and Anton van den Hengel*

### Proposed Model

#### Question Embedding
* Embedding: 300 dimension, initialized with public Glove(or zero vector if words not exist), end_padded
* Max_len: 14, no trim, no START token, no END token
* RNN module: GRU, hidden state dimension 512

#### Image Features
* $k \times 2048$, two ways to extract image features:
  1. 200-layer ResNet, features maps: $14 \times 14 \times 2048$, followed by a $2 \times 2$ pooling to get a $7 \times 7 \times 2048$ feature maps, $K = 49$
  2. R-CNN, $K=36$
 
#### Image Attention(top-down attention)
Each image feature vector $\nu_i$ is first concatenated with question embedding vector $\mathbf{q}$. Then they are passed through a non-linear layer and a linear layer to obtain the attention weight. An averaged image featues could be obtained with this attention weights. According to the paper, this averaged image feature is called *question-guided attention*.

\begin{equation}
\begin{aligned}
a_i &= \mathbf{\omega}_af_a([\mathbf{\nu}_i, \mathbf{q}])\\
\mathbf{\alpha} &= softmax(\mathbf{a})\\
\hat{\mathbf{v}} &= \sum_{i=1}^{K}\alpha_i\mathbf{\nu}_i\\
\end{aligned}
\end{equation}

#### Multimodal Fusion
Both question embedding vector $\mathbf{q}$ and question-quided attention are passed through non-linear layers seperately and then combine them with a simple Hadamard product. The output is called joint embedding of the question and of the image.
$$
\mathbf{h} = f_q(\mathbf{q}) \circ f_\nu(\hat{\nu})
$$

#### Output classifier
Joint embedding $\mathbf{h}$ is first passed through a non-linear layer and then a linear layer. A sigmoid activation function is applied to the output of the linear layer.
$$
\hat{s} = \sigma(\mathbf{\omega}_o \  f_o(\mathbf{h}))
$$
The loss function of this classifier is one of the most interesting part of this paper. It uses a *soft* target scores. Unlike other models, the groundtruth is a one-hot vector with only a 'one' at the index which corresponds to the correct answer and the rest of the vector are all zeros. The *soft* target scores has values between $(0, 1)$ at all locations and could be interpret as the level of certainty for each answer. I think this is true for the VQA problem because there is disagreement between human annotators.
$$
L = -\sum_{i}^{M}\sum_{j}^{N}\big(s_{ij}\log(\hat{s}_{ij}) - (1-s_{ij})\log(1-\hat{s}_{ij})\big)
$$

#### Pretraining the classifier
The weight matrix in the output stage is initialized as follows:
* Texts: Using GloVe embedding of the answers at corresponding position of the matrix, or zero vectors if words not exist in Glove
* Images: Take average feature using RexNet-101 CNN for 10 relevant images googled from internet
* Due to different sizes of embedding feature and image feature, they are first passed through a non-linear layer:
$$
\hat{s} = \sigma\big(\omega_o^{text} f_o^{text}(\mathbf{h}) + \omega_o^{img}f_o^{img}(\mathbf{h})\big)
$$

*Question*(*TODO: not clear*) I have a question about this trick. At the beginning, questions and images are mapped into their embedding space using Glove and RexNet, respectily. The weight matrix, again, is initialized with this infomation. Then, like calculating similarity, most similar words or images will have higher value. In this procedure, the answers' corresponding image features are hard coded into the system. I think even if we remove the GRU model, the result will still remain good.

*Section 4.6*And from the paper, thay said, this trick has better recall on 

#### Non-linear layers
They used a gated hyperbolic tangent activation. They borrowed the 'gate' idea from LSTMs and GRUs. This is another interesing part to me.
\begin{equation}
\begin{aligned}
\mathbf{\tilde{y}} &= \tanh(Wx + b)\\
\mathbf{g} &= \sigma(W'x + b')\\
\mathbf{y} &= \mathbf{\tilde{y}} \circ \mathbf{g}\\
\end{aligned}
\end{equation}



