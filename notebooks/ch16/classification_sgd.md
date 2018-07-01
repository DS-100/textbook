

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
np.random.seed(42)
```

## Gradient Descent in Logistic Regression

Previously, we covered batch gradient descent, an algorithm that iteratively updates $\hat\theta$ to find the loss-minimizing parameters. We also discussed stochastic gradient descent and mini-batch gradient descent, methods that take advantage of statistical theory and parallelized hardware to decrease the time spent training the gradient descent algorithm. In this section, we will apply these concepts to logistic regression and walk through examples using scikit-learn functions.

### Batch Gradient Descent

The general update formula for batch gradient descent is given by:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \cdot \nabla_\hat\theta L(\hat\theta, X, y)
$$

In logistic regression, we use the cross entropy loss as our loss function:

$$
L(\hat\theta, X, y) = \frac{1}{n} \sum_{i} \left(-y_i \ln \left(f_{\hat\theta} \left(X_i \right) \right) - \left(1 - y_i \right) \ln \left(1 - f_{\hat\theta} \left(X_i \right) \right) \right)
$$

$\nabla_{\hat\theta} L(\hat\theta, X, y) = -\frac{1}{n}\sum_{i=1}^n(y_i - \sigma_i)X_i $ is then the gradient of the cross entropy loss; plugging this in allows us to find the gradient descent algorithm specific to logistic regression. Letting $ \sigma_i = f_\hat\theta(X_i) = \sigma(X_i \cdot \hat \theta) $, this becomes:

$$
\begin{align}
\hat\theta_{t+1} &= \hat\theta_t - \alpha \cdot \left(- \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) X_i \right) \\
&= \hat\theta_t + \alpha \cdot \left(\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) X_i \right)
\end{align}
$$

- $\hat\theta_t$ is the current estimate of $\theta$ at iteration $t$
- $\alpha$ is the learning rate
- $-\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) X_i$ is the gradient of the cross entropy loss
- $\hat\theta_{t+1}$ is the next estimate of $\theta$ computed by subtracting the product of $\alpha$ and the cross entropy loss computed at $\hat\theta_t$


### Stochastic Gradient Descent

The general update formula is below, where $l(\hat\theta, X_i, y_i)$ is the loss function for a single data point:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \nabla_\hat\theta l(\hat\theta, X_i, y_i)
$$

Returning back to our example in logistic regression, we approximate the gradient of the cross entropy loss across all data points using the gradient of the cross entropy loss of one data point. This is shown below, with $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $.

$$
\begin{align}
\nabla_\hat\theta L(\hat\theta, X, y) &\approx \nabla_\hat\theta l(\hat\theta, X_i, y_i)\\
&= -(y_i - \sigma_i)X_i
\end{align}
$$

When we plug this approximation into the general formula for stochastic gradient descent, we find the stochastic gradient descent update formula for logistic regression.

$$
\begin{align}
\hat\theta_{t+1} &= \hat\theta_t - \alpha \nabla_\hat\theta l(\hat\theta, X_i, y_i) \\
&= \hat\theta_t + \alpha \cdot (y_i - \sigma_i)X_i
\end{align}
$$

### Mini-batch Gradient Descent

Similarly, we can approximate the gradient of the cross entropy loss using a random sample of data points, also known as a mini-batch.

$$
\nabla_\hat\theta L(\hat\theta, X, y) \approx \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\nabla_{\hat\theta}l(\hat\theta, X_i, y_i)
$$

We substitute this for the gradient of the cross entropy loss, yielding a mini-batch gradient descent update formula specific to logistic regression:

$$
\begin{align}
\hat\theta_{t+1} &= \hat\theta_t - \alpha \cdot -\frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)X_i \\
&= \hat\theta_t + \alpha \cdot \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)X_i
\end{align}
$$

## Implementation in Scikit-Learn

Scikit-learn has implementations for stochastic gradient descent and mini-batch gradient descent using the [`SGDClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) class. To gain a better estimate of the speed benefits afforded by stochastic gradient descent and mini-batch gradient descent, we will first manually implement batch gradient descent; then, we will compare the results with stochastic gradient descent and mini-batch gradient descent.


```python
lebron = pd.read_csv('lebron.csv')

columns = ['shot_distance', 'minute', 'action_type', 'shot_type', 'opponent']
rows = lebron[columns].to_dict(orient='row')

onehot = DictVectorizer(sparse=False).fit(rows)
X = onehot.transform(rows)
y = lebron['shot_made'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, random_state=42
)
```


```python
# from sklearn.datasets import load_iris

# iris = load_iris()
# X = iris['data']
# y = iris['target'] == 2

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=20, random_state=42
# )
```


```python
%%time
# Stochastic GD
sgd_clf = SGDClassifier(loss='log', random_state=42)
sgd_clf.fit(X_train, y_train)
```

    CPU times: user 0 ns, sys: 15.6 ms, total: 15.6 ms
    Wall time: 1.77 ms



```python
sgd_clf.score(X_test, y_test)
```




    0.5




```python
# Mini-batch GD

def iter_minibatches(X, y, minibatch_size):
    # Provide chunks one by one
    shuffled_indices = np.random.permutation(len(y))
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    for i in range(0, len(y), minibatch_size):
        X_b, y_b = X[i:i+minibatch_size], y[i:i+minibatch_size]
        yield X_b, y_b
        
minibatch_generator = iter_minibatches(X_train, y_train, 10)
```


```python
%%time
mbgd_clf = SGDClassifier(loss='log', random_state=42)
for X_b, y_b in minibatch_generator:
    mbgd_clf.partial_fit(X_b, y_b, classes=np.unique(y))
```

    CPU times: user 15.6 ms, sys: 0 ns, total: 15.6 ms
    Wall time: 16.8 ms



```python
mbgd_clf.score(X_test, y_test)
```




    0.5


