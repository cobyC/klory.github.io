---
title: Summary on attention used in VQA 
---

#### Study note on Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering

A Multi-modal Factorized Bilinear Pooling(MFB) approach is developed to achieve more effective fusion of viusal features and textual features.
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
 z/\|z\|)$ normalization layers are appended after MFB output.
* Relation to MLB: according to the author, MLB is a special case of MFB with $k = 1$.

#### Hierarchical Question-Image Co-Attention for Visual Question Answering
* Extract hierarchical question features
\begin{equation}
\begin{aligned}
\mathbf{\hat{q}}_{s,t}^p &= tanh(\mathbf{W}_c^s \mathbf{q_{t:t+s-1}^w)}, s \in {1, 2, 3}\\
\mathbf{\hat{q}} &= max(\mathbf{\hat{q}}_{1,t}^p, \mathbf{\hat{q}}_{2,t}^p, \mathbf{\hat{q}}_{3,t}^p )\\
\end{aligned}
\end{equation}

* Parallel Co-Attention
\begin{equation}
\begin{aligned}
\mathbf{C} &= tanh(\mathbf{Q}^T\mathbf{W}_b\mathbf{V})\\
\mathbf{H}^v &= tanh(\mathbf{W}_v\mathbf{V} + (\mathbf{W}_q\mathbf{Q})\mathbf{C})\\
\mathbf{H}^q &= tanh(\mathbf{W}_q\mathbf{Q} + (\mathbf{W}_v\mathbf{V})\mathbf{C})\\
\mathbf{a}^v &= softmax(\mathbf{w}_{hv}^T\mathbf{H}^v)\\
\mathbf{a}^q &= softmax(\mathbf{w}_{hq}^T\mathbf{H}^q)\\
\mathbf{\hat{v}} &= \sum_{n=1}^{N} a_n^v\mathbf{v}_n\\
\mathbf{\hat{q}} &= \sum_{t=1}^{T} a_t^q\mathbf{q}_t\\
\end{aligned}
\end{equation}

The parallel Co-attention is done at each level in hierarchy, so, at last, we will get $\mathbf{v}^w, \mathbf{v}^p, \mathbf{v}^s$ and $\mathbf{q}^w, \mathbf{q}^p, \mathbf{q}^s$.
* Alternating Co-Attention
\begin{equation}
\begin{aligned}
\mathbf{H} &= tanh(\mathbf{W}_x\mathbf{X} + (\mathbf{W}_g\mathbf{g})\mathbf{1}^T)\\
\mathbf{a}^x &= softmax(\mathbf{w}_{hx}^T\mathbf{H})\\
\mathbf{\hat{x}} &= \sum a_i^x\mathbf{x}_i\\
\end{aligned}
\end{equation}

At the beginning, $\mathbf{X}$ is set to $\mathbf{Q}$, question features, and $\mathbf{g}$ is set to $0$. And then $\mathbf{X}$ is set to $\mathbf{V}$ and $\mathbf{g}$ is set to the intermidiate attended question feature calculated from the first step.

From the result of this paper, Alternating Co-attention has a better performance than Parallel Co-Attention.

* Prediction
\begin{equation}
\begin{aligned}
\mathbf{h}^w &= tanh(\mathbf{w}_w(\mathbf{\hat{q}}^w + \mathbf{\hat{v}}^w))\\
\mathbf{h}^p &= tanh(\mathbf{w}_p[(\mathbf{\hat{q}}^p + \mathbf{\hat{v}^p}), \mathbf{\hat{h}}^w])\\
\mathbf{h}^s &= tanh(\mathbf{w}_s[(\mathbf{\hat{q}}^s + \mathbf{\hat{v}^s}), \mathbf{\hat{h}}^p])\\
\end{aligned}
\end{equation}

