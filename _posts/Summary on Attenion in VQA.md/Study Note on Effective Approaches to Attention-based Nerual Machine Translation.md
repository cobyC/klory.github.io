---
title: Study Note on Effective Approaches to Attention-based Nerual Machine Translation
---

*Minh-Thang Luong, Hieu Pham, and Christopher D.Manning*

A neural machine translation system directly model the conditional probability $p(y|x)$ of translating a source sentence, $x_1, \dots, x_n$ to a target sentence, $y_1, \dots, y_n$.

Two types of attention-bases models are proposed in this paper:
1. Global Attention:
* Attention is placed on all source positions.
* Alignment vector $a_t$ is computed as: 
$$
\begin{equation}
\begin{aligned}
a_t(s) &= align(\mathbf{h}_t, \mathbf{\bar{h}}_s)\\
&= \frac{\exp(score(\mathbf{h}_t, \mathbf{\bar{h}}_s))}{\sum_{s'}\exp(score(\mathbf{h}_t, \mathbf{\bar{h}}_{s'}))}
\end{aligned}
\end{equation}
$$
* Context vector is computed as the weighted average over all the source hidden states.
* To my understanding, the form of the *score* function is quite flexible, as long as it's chosen to be able to measure *distance* or *similarity* between two vectots.
2. Local Atttention
* Attention is placed on only a few source positions.

