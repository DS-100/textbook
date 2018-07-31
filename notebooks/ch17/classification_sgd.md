

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
np.random.seed(42)
```

## Fitting a Logistic Model

Previously, we covered batch gradient descent, an algorithm that iteratively updates $\boldsymbol{\theta}$ to find the loss-minimizing parameters $\boldsymbol{\hat\theta}$. We also discussed stochastic gradient descent and mini-batch gradient descent, methods that take advantage of statistical theory and parallelized hardware to decrease the time spent training the gradient descent algorithm. In this section, we will apply these concepts to logistic regression and walk through examples using scikit-learn functions.

### Batch Gradient Descent

The general update formula for batch gradient descent is given by:

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \cdot \nabla_\boldsymbol{\theta} L(\boldsymbol{\theta}^{(t)}, \boldsymbol{X}, \boldsymbol{y})
$$

In logistic regression, we use the cross entropy loss as our loss function:

$$
L(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) = \frac{1}{n} \sum_{i=1}^{n} \left(-y_i \ln \left(f_{\boldsymbol{\theta}} \left(\boldsymbol{X}_i \right) \right) - \left(1 - y_i \right) \ln \left(1 - f_{\boldsymbol{\theta}} \left(\boldsymbol{X}_i \right) \right) \right)
$$

The gradient of the cross entropy loss is $\nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) = -\frac{1}{n}\sum_{i=1}^n(y_i - \sigma_i)\boldsymbol{X}_i $. Plugging this into the update formula allows us to find the gradient descent algorithm specific to logistic regression. Letting $ \sigma_i = f_\boldsymbol{\theta}(\boldsymbol{X}_i) = \sigma(\boldsymbol{X}_i \cdot \boldsymbol{\theta}) $:

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)} &= \boldsymbol{\theta}^{(t)} - \alpha \cdot \left(- \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) \boldsymbol{X}_i \right) \\
&= \boldsymbol{\theta}^{(t)} + \alpha \cdot \left(\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) \boldsymbol{X}_i \right)
\end{align}
$$

- $\boldsymbol{\theta}^{(t)}$ is the current estimate of $\boldsymbol{\theta}$ at iteration $t$
- $\alpha$ is the learning rate
- $-\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) \boldsymbol{X}_i$ is the gradient of the cross entropy loss
- $\boldsymbol{\theta}^{(t+1)}$ is the next estimate of $\boldsymbol{\theta}$ computed by subtracting the product of $\alpha$ and the cross entropy loss computed at $\boldsymbol{\theta}^{(t)}$


### Stochastic Gradient Descent

Stochastic gradient descent approximates the gradient of the loss function across all observations using the gradient of the loss of a single data point.The general update formula is below, where $\ell(\boldsymbol{\theta}, \boldsymbol{X}_i, y_i)$ is the loss function for a single data point:

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \nabla_\boldsymbol{\theta} \ell(\boldsymbol{\theta}, \boldsymbol{X}_i, y_i)
$$

Returning back to our example in logistic regression, we approximate the gradient of the cross entropy loss across all data points using the gradient of the cross entropy loss of one data point. This is shown below, with $ \sigma_i = f_{\boldsymbol{\theta}}(\boldsymbol{X}_i) = \sigma(\boldsymbol{X}_i \cdot \boldsymbol{\theta}) $.

$$
\begin{align}
\nabla_\boldsymbol{\theta} L(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) &\approx \nabla_\boldsymbol{\theta} \ell(\boldsymbol{\theta}, \boldsymbol{X}_i, y_i)\\
&= -(y_i - \sigma_i)\boldsymbol{X}_i
\end{align}
$$

When we plug this approximation into the general formula for stochastic gradient descent, we find the stochastic gradient descent update formula for logistic regression.

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)} &= \boldsymbol{\theta}^{(t)} - \alpha \nabla_\boldsymbol{\theta} \ell(\boldsymbol{\theta}, \boldsymbol{X}_i, y_i) \\
&= \boldsymbol{\theta}^{(t)} + \alpha \cdot (y_i - \sigma_i)\boldsymbol{X}_i
\end{align}
$$

### Mini-batch Gradient Descent

Similarly, we can approximate the gradient of the cross entropy loss for all observations using a random sample of data points, known as a mini-batch.

$$
\nabla_\boldsymbol{\theta} L(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) \approx \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}, \boldsymbol{X}_i, y_i)
$$

We substitute this approximation for the gradient of the cross entropy loss, yielding a mini-batch gradient descent update formula specific to logistic regression:

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)} &= \boldsymbol{\theta}^{(t)} - \alpha \cdot -\frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)\boldsymbol{X}_i \\
&= \boldsymbol{\theta}^{(t)} + \alpha \cdot \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)\boldsymbol{X}_i
\end{align}
$$

## Implementation in Scikit-learn

Scikit-learn's [`SGDClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) class provides an implementation for stochastic gradient descent, which we can use by specifying `loss=log`. Since scikit-learn does not have a model that implements batch gradient descent, we will compare `SGDClassifier`'s performance against [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) on the `emails` dataset. We omit feature extraction for brevity:


```python
# HIDDEN
emails = pd.read_csv('emails_sgd.csv').sample(frac=0.5)

X, y = emails['email'], emails['spam']
X_tr = CountVectorizer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tr, y, random_state=42)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
```


```python
log_reg = LogisticRegression(tol=0.0001, random_state=42)
stochastic_gd = SGDClassifier(tol=0.0001, loss='log', random_state=42)
```


```python
%%time
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print('Logistic Regression')
print('  Accuracy:  ', accuracy_score(y_test, log_reg_pred))
print('  Precision: ', precision_score(y_test, log_reg_pred))
print('  Recall:    ', recall_score(y_test, log_reg_pred))
print()
```

    Logistic Regression
      Accuracy:   0.9913793103448276
      Precision:  0.974169741697417
      Recall:     0.9924812030075187
    
    CPU times: user 3.2 s, sys: 0 ns, total: 3.2 s
    Wall time: 3.26 s



```python
%%time
stochastic_gd.fit(X_train, y_train)
stochastic_gd_pred = stochastic_gd.predict(X_test)
print('Stochastic GD')
print('  Accuracy:  ', accuracy_score(y_test, stochastic_gd_pred))
print('  Precision: ', precision_score(y_test, stochastic_gd_pred))
print('  Recall:    ', recall_score(y_test, stochastic_gd_pred))
print()
```

    Stochastic GD
      Accuracy:   0.9808429118773946
      Precision:  0.9392857142857143
      Recall:     0.9887218045112782
    
    CPU times: user 93.8 ms, sys: 31.2 ms, total: 125 ms
    Wall time: 119 ms


The results above indicate that `SGDClassifier` is able to find a solution in significantly less time than `LogisticRegression`. Although the evaluation metrics are slightly worse on the `SGDClassifier`, we can improve the `SGDClassifier`'s performance by tuning hyperparameters. Furthermore, this discrepancy is a tradeoff that data scientists often encounter in the real world. Depending on the situation, data scientists might place greater value on the lower runtime or on the higher metrics.

## Summary

Stochastic gradient descent is a method that data scientists use to cut down on computational cost and runtime. We can see the value of stochastic gradient descent in logistic regression, since we would only have to calculate the gradient of the cross entropy loss for one observation at each iteration instead of for every observation in batch gradient descent. From the example using scikit-learn's `SGDClassifier`, we observe that stochastic gradient descent may achieve slightly worse evaluation metrics, but drastically improves runtime. On larger datasets or for more complex models, the difference in runtime might be much larger and thus more valuable.
