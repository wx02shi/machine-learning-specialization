---
tags: []
aliases: []
---
$$\begin{align*}
\frac{\delta}{\delta w}J(w,b) &= \frac{\delta}{\delta w} \frac{1}{2m}\sum_{i=1}^m  \left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)^2 \\
&= \frac{\delta}{\delta w} \frac{1}{2m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right)^2 \\
&= \frac{1}{2m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right) 2x^{(i)} \\
&= \frac{1}{m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right) x^{(i)}
\end{align*}$$

This is why we defined the cost function to divide by $2m$ instead of $m$, since it makes the math easier.

$$\begin{align*}
\frac{\delta}{\delta b}J(w,b) &= \frac{\delta}{\delta b} \frac{1}{2m}\sum_{i=1}^m  \left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)^2 \\
&= \frac{\delta}{\delta b} \frac{1}{2m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right)^2 \\
&= \frac{1}{2m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right) 2 \\
&= \frac{1}{m}\sum_{i=1}^m  \left(wx^{(i)}+b-y^{(i)}\right)
\end{align*}$$
