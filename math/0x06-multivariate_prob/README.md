Resources
Read or watch:

Joint Probability Distributions
Multivariate Gaussian distributions
The Multivariate Gaussian Distribution
An Introduction to Variance, Covariance & Correlation
Variance-covariance matrix using matrix notation of factor analysis
Definitions to skim:

Carl Friedrich Gauss
Joint probability distribution
Covariance
Covariance matrix
As references:

numpy.cov
numpy.corrcoef
numpy.linalg.det
numpy.linalg.inv
numpy.random.multivariate_normal
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
Who is Carl Friedrich Gauss?
What is a joint/multivariate distribution?
What is a covariance?
What is a correlation coefficient?
What is a covariance matrix?
What is a multivariate Gaussian distribution?


Edge case
```
#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

np.random.seed(6)
X = np.random.multivariate_normal([5, -4, 2], [[6, -3, 5], [-3, 10, -2], [5, -2, 5]], 10000).T
mn = MultiNormal(X)
x = X[:, 100:101]
print(mn.pdf(x))
```