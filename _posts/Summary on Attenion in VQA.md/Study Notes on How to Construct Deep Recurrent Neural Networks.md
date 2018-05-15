---
title: Study Notes on How to Construct Deep Recurrent Neural Networks
---
*Razvan Pascanu, Caglar Gulceher, Kyunghyun Cho, and Yoshua Bengio*


The definetion of **depth** in MLP does not apply trivially to a recurrent neural network because of its temporal structure.
A close analysis of the computation at each time step shows that the transitions in RNN, input-to-hidden, hidden-to-hidden, hidden-to-output, are not deep, but are only results of a linear projection followed by an element-wise nonlinearity.

Three modification are proposed in this paper:
* Deep Input-to-Hidden Function: Depth makes it easier to learn temporal structure between successive time steps, because abstract features can generally be expressed more easily.
* Deep Hidden-to-Output Function: Depth can be useful to disentangle the factors of variations in the hidden stat, making it easier to predict the output.
* Deep Hidden-to-Hidden Transition: Depth allow hidden state of RNN guickly changing modes of input. The problem of difficulty in training this kind networks due to gradient traversing among multiple layers could be addressed by introducing shortcut connections between layers.