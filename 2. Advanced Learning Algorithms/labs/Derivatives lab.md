---
tags: [lab]
alias: []
---
```python
from sympy import symbols, diff
```

# Finding symbolic derivatives
The process of differentiation has been automated with symbolic differentiation programs like SymPy. 
Let's use the example:
$$J=w^2$$
Define the python variables and their symbolic names.
```python
J, w = symbols('J, w')
```
Define and print the expression. Note SymPy produces a LaTeX string which generates a nicely readable equation. 
```python
J=w**2
J
```
$w^2$
Use `diff` to differentiate the expression for $J$ with respect to $w$. 
```python
DJ_dw = diff(J,w)
DJ_dw
```
$2w$
Evaluate the derivative at a few poitns by 'substituting' numeric values for the symbolic values.
```python
dJ_dw.subs([(w,2)])
dJ_dw.subs([(w,3)])
dJ_dw.subs([(w,-3)])
```
$4$
$6$
$-6$
