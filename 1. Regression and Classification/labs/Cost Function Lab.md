---
tags: [lab]
alias: []
---
Note that a lot of the Python code provided in the original lab is for providing the necessary components to display graphs. 
At the moment, I don't think I really care about being able to see my soup bowl or contour plot...

Tools:
```Python
import numpy as np
```

Define a function that computes the cost of using values $w$ and $b$:
```Python
def compute_cost(x, y, w, b):
	m = x.shape[0]

	cost_sum = 0
	for i in range(m):
		f_wb = w * x[i] + b
		cost = (f_wb - y[i]) ** 2
		cost_sum = cost_sum + cost
	total_cost = (1 / (2*m)) * cost_sum

	return total_cost
```
