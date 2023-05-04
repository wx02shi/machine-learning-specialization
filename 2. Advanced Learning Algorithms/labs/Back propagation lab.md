---
tags: [lab]
alias: []
---
Back propagation using a computation graph

```python
from sympy import *
import numpy as np
import re
%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *
```

# Computation graph
A computation graph simplifies the computation of complex derivatives by breaking them into smaller steps.
Let's calculate the derivative fo $J=(2+3w)^2$.

## Forward propagation
```python
w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")
```
```
a = 11, J = 121
```

## Backprop
```python
sw, sJ, sa = symbols('w,J,a')
sJ = sa**2
sJ
```
$a^2$
```python
sJ.subs([sa,a])
```
$121$
```python
dJ_da = diff(Sj, sa)
DJ_da
```
$2a$
```python
sa = 2 + 3*sw
sa
```
$3w+2$
```python
da_dw = diff(sa,sw)
da_dw
```
$3$
```python
dJ_dw = da_dw * DJ_da
dJ_dw
```
$6a$

# Computation graph of a simple neural network
![[Pasted image 20230128195216.png]]
## Forward prop
```python
x = 2
w = -2
b = 8
y = 1

c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")
```
```
J=4.5, d=3, a=4, c=-4
```

## Backprop
```python
sx, sw, sb, sy, sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
sJ
```
$\frac{d^2}{2}$
```python
sJ.subs([(sd,d)])
```
$\frac{9}{2}$
```python
dJ_dd = diff(sJ, sd)
dJ_dd
```
$d$
```python
sd = sa - sy
sd
```
$a-y$
```python
dd_da = diff(sd, sa)
dd_da
```
$1$
```python
dJ_da = dd_da * dJ_dd
dJ_da
```
$d$
```python
sa = sc + sb
sa
```
$b+c$
```python
da_dc = diff(sa, sc)
da_db = diff(sa, sb)
print(da_dc, da_db)
```
```
1 1
```
```python
dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")
```
```
dJ_dc = d,  dJ_db = d
```

```python
sc = sw*sx
sc
```
$wx$
```python
dJ_dw = dc_dw * dJ_dc
dJ_dw
```
$dx$
```python
print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")
```
```
dJ_dw = 2*d
```