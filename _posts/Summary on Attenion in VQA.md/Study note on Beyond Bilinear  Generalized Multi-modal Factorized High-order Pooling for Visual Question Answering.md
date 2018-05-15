---
title: Study note on Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering
---

*Zhou Yu, Jun Yu, Chenchao Xiang, Jianping Fan, and Dacheng Tao*

Three issues should be solved effectively:
* Extracting discriminative features for image and question representations,
* Combining the visual features from the image and textual features from question to generate the fused image-question features,
* Using the fused image-question features to learn a multi-class classifier for predicting the best matching answer correctly.

The paper's contibution:
1. A Multi-modal Factorized Bilinear Pooling(MFB) approach is developed to achieve more effective fusion of viusal features and textual features.
* Start from multi-modal bilinear model and factorize matrix $W_i$:
\begin{equation}
\begin{aligned}
z_i &= x^{T}W_iy\\
&= x^{T}U_iV_i^{T}y = \sum^{k}_{d=1}\\
&= \mathbb{1}^{T}(U_i^{T}x \circ V_i^{T}y)
\end{aligned}
\end{equation}
where $k$ is the factor or the latent dimentsionality of the factorized matrix. To obtain the output feature $z \in \mathbb R^o$, we need two three-order tensors $U = [U_1, \dots, U_O] \in \mathbb R^{m\times k \times o}$ and $V = [V_1, \dots, V_o] \in \mathbb V^{n\times k \times o}$
* Reformulate $U$ and $V$: $\tilde{U} \in \mathbb{R}^{m \times ko}$ and $\tilde{V} \in \mathbb{R}^{n \times ko}$
$$
z = SumPool(\tilde{U}^{T}x \circ \tilde{V}^{T}y, k)
$$
where the function $SumPool(x, k)$ means using a one-dimensional non-overlapped window with the size k to perform sum pooling over x.
* The power normalization($z \gets sign(z)|z|^{0.5}$) and $l2(z \gets z/\|z\|)$ normalization layers are appended after MFB output.
* Relation to MLB: according to the author, MLB is a special case of MFB with $k = 1$. *To my understanding, they are basically the same accorind to the matrix mulplication and summation step.*

2. A generalized Multi-modal Factorized High-order pooling approach is developed by cascading multiple MFB blocks.
* MFB can be seperated into expand stage and the squeeze state:
\begin{equation}
\begin{aligned}
z_{exp} &= MFB_{exp}(x, y) = Dropout(\tilde{U}^{T}x \circ \tilde{V}^{T}y) \in \mathbb{R}^{ko}\\
z &= MFB_{sqz}(z_{exp}) = Norm(SumPool(z_{exp})) \in \mathbb{R}^{o}
\end{aligned}
\end{equation}
* Make $p$ MFB blocks cascadable:
\begin{equation}
\begin{aligned}
z_{exp}^i &= MFB_{exp}^i(x, y) = z_{exp}^{i-1} \circ (Dropout(\tilde{U}^{i^T}x \circ \tilde{V}^{i^T}y))\\
z &= MFH^p = [z^1, z^2, \dots, z^p] \in \mathbb{R}^{op}
\end{aligned}
\end{equation}
3. Aco-attention learning architecture is designed:
* Both the image attention module and question attention module consist of sequentiao $1 \times 1$ convolutionsal layers and ReLU layers followed by the softmax normalization layers to predict the attention weights for each input feature.
* The question attention is learned in a self-attention manner.
* Question attentive representation is fed into an image attention module.
4. The KL divergence is used as loss function.
$$
L = \sum_{i}y_i\log(\frac{y_i}{z_i}) = - KL(\hat{p}\|p)
$$
where $\hat{p}$ is the eatimated probability.