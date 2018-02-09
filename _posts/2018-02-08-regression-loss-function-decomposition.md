---
title: Decomposition of Loss functions for regression
---
In PRML, the loss function for regression is (1.86)

$$E[L]=\int\int L(t, y(\mathbf{x})p(\mathbf{x}, t) d \mathbf{x} dt$$

And it can be decomposed into (1.90)

$$E[L] = \int \{y(x) - E[t|x]\}^2 p(x)dx + \int var[t|x]p(x)dx$$

And below is the proof.
![Proof](https://github.com/klory/klory.github.io/blob/master/images/loss_function_decomposition.jpg?raw=true)
