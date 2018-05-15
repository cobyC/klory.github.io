---
title: Study None on Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
---

*Kyunghyun Cho, Bart van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Dzmitry Bahdanau, Holger Schwenk, and Yoshua Bengio*

The main part that I focus on this paper the the proposed GRU model. The main difference between a LSTM and a GRU is how the hidden state is updated:

At time step $T = t$, for hidden unit $j$:

\begin{equation}
\begin{aligned}
r_j &= \sigma([\mathbf{W}_rx]_j + [\mathbf{U}_rh_{<t-1>}]_j)\\
z_j &= \sigma([\mathbf{W}_zx]_j + [\mathbf{U}_zh_{<t-1>}]_j)\\
h^{t}_j &= z_jh_j^{t-1} + (1-z_j)\hat{h}^{t}_j\\
\hat{h}^{t}_j &= \phi([\mathbf{W}x]_j + [\mathbf{U}(\mathbf{r} \odot h_{<t-1>})]_j)
\end{aligned}
\end{equation}

#TODO 
* add up LSTM code
* add up comparison and conclusion

