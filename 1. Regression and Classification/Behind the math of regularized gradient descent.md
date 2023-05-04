---
tags: []
alias: []
---
We can rearrange the update expression for $w_j$

$$w_j'=w_j\left(1-\alpha\frac{\lambda}{m}\right)-\alpha \frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}$$
The term $(1-\alpha\frac{\lambda}{m})$ is the interesting part. When we choose smaller values for $\alpha$ and $\lambda$, we get this entire term to be close to 1, but not exactly 1. Effectively, upon every iteration of gradient descent, we're forcing $w_j$ to get a little bit smaller!

For example, $\alpha=0.01$ and $\lambda=1$ gives us $w_j(1-0.02)=w_j\times 0.9998$.