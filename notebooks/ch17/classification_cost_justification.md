

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```

## Approximating the Empirical Probability Distribution

In this section, we introduce **KL divergence** and demonstrate how minimizing average KL divergence in binary classification is equivalent to minimizing average cross-entropy loss.

Since logistic regression outputs probabilities, a logistic model produces a certain type of probability distribution. Specifically, based on optimal parameters $ \hat{\boldsymbol{\theta}} $, it estimates the probability that the label $ y $ is $ 1 $ for an example input $ \textbf{x} $.

For example, suppose that $ x $ is a scalar recording the forecasted chance of rain for the day and $ y = 1 $ means that Mr. Doe takes his umbrella with him to work. A logistic model with scalar parameter $ \hat{\theta} $ predicts the probability that Mr. Doe takes his umbrella given a forecasted chance of rain: $ \hat{P_\theta}(y = 1 | x) $.

Collecting data on Mr. Doe's umbrella usage provides us with a method of constructing an empirical probability distribution $ P(y = 1 | x) $. For example, if there were five days where the chance of rain $ x = 0.60 $ and Mr. Doe only took his umbrella to work once, $ P(y = 1 | x = 0.60) = 0.20 $. We can compute a similar probability distribution for each value of $ x $ that appears in our data. Naturally, after fitting a logistic model we would like the distribution predicted by the model to be as close as possible to the empirical distribution from the dataset. That is, for all values of $ x $ that appear in our data, we want:

$$ \hat{P_\theta}(y = 1 | x) \approx P(y = 1 | x) $$

One commonly used metric to determine the "closeness" of two probability distributions is the Kullback–Leibler divergence, or KL divergence, which has its roots in information theory.

## Defining Average KL Divergence

KL divergence quantifies the difference between the probability distribution $\hat{P_\boldsymbol{\theta}}$ computed by our logistic model with parameters $ \boldsymbol{\theta} $ and the actual distribution $ P $ based on the dataset. Intuitively, it calculates how imprecisely the logistic model estimates the distribution of labels in data.

The KL divergence for binary classification between two distributions $P$ and $\hat{P_\boldsymbol{\theta}}$ for a single data point $(\textbf{x}, y)$ is given by:

$$D(P || \hat{P_\boldsymbol{\theta}}) = P(y = 0 | \textbf{x}) \ln \left(\frac{P(y = 0 | \textbf{x})}{\hat{P_\boldsymbol{\theta}}(y = 0 | \textbf{x})}\right) + P(y = 1 | \textbf{x}) \ln \left(\frac{P(y = 1 | \textbf{x})}{\hat{P_\boldsymbol{\theta}}(y = 1 | \textbf{x})}\right)$$

KL divergence is not symmetric, i.e., the divergence of $\hat{P_\boldsymbol{\theta}}$ from $P$ is not the same as the divergence of $P$ from $\hat{P_\boldsymbol{\theta}}$: $$D(P || \hat{P_\boldsymbol{\theta}}) \neq D(\hat{P_\boldsymbol{\theta}} || P)$$  

Since our goal is to use $\hat{P_\boldsymbol{\theta}}$ to approximate $P$, we are concerned with $ D(P || \hat{P_\boldsymbol{\theta}}) $.

The best $\boldsymbol{\theta}$ values, which we denote as $\hat{\boldsymbol{\theta}}$, minimize the average KL divergence of the entire dataset of $n$ points:

$$ \text{Average KL Divergence} = \frac{1}{n} \sum_{i=1}^{n} \left(P(y_i = 0 | \textbf{X}_i) \ln \left(\frac{P(y_i = 0 | \textbf{X}_i)}{\hat{P_\boldsymbol{\theta}}(y_i = 0 | \textbf{X}_i)}\right) + P(y_i = 1 | \textbf{X}_i) \ln \left(\frac{P(y_i = 1 | \textbf{X}_i)}{\hat{P_\boldsymbol{\theta}}(y_i = 1 | \textbf{X}_i)}\right)\right)$$
$$ \hat{\boldsymbol{\theta}} = \displaystyle\arg \min_{\substack{\boldsymbol{\theta}}} (\text{Average KL Divergence}) $$

In the above equation, the $i^{\text{th}}$ data point is represented as ($ \textbf{X}_i $, $ y_i $) where $ \textbf{X}_i $ is the $i^{\text{th}}$ row of the $n \times p$ data matrix $\textbf{X}$ and $ y_i $ is the observed outcome.

KL divergence does not penalize mismatch for rare events with respect to $P$. If the model predicts a high probability for an event that is actually rare, then both $P(k)$ and $\ln \left(\frac{P(k)}{\hat{P_\boldsymbol{\theta}}(k)}\right)$ are low so the divergence is also low. However, if the model predicts a low probability for an event that is actually common, then the divergence is high. We can deduce that a logistic model that accurately predicts common events has a lower divergence from $P$ than does a model that accurately predicts rare events but varies widely on common events. 

## Deriving Cross-Entropy Loss from KL Divergence

The structure of the above average KL divergence equation contains some surface similarities with cross-entropy loss. We will now show with some algebraic manipulation that minimizing average KL divergence is in fact equivalent to minimizing average cross-entropy loss.

Using properties of logarithms, we can rewrite the weighted log ratio:
$$P(y_i = k | \textbf{X}_i) \ln \left(\frac{P(y_i = k | \textbf{X}_i)}{\hat{P_\boldsymbol{\theta}}(y_i = k | \textbf{X}_i)}\right) = P(y_i = k | \textbf{X}_i) \ln P(y_i = k | \textbf{X}_i) - P(y_i = k | \textbf{X}_i) \ln \hat{P_\boldsymbol{\theta}}(y_i = k | \textbf{X}_i)$$

Note that since the first term doesn't depend on $\boldsymbol{\theta}$, it doesn't affect $\displaystyle\arg \min_{\substack{\boldsymbol{\theta}}}$ and can be removed from the equation. The resulting expression is the cross-entropy loss of the model $\hat{P_\boldsymbol{\theta}}$:

$$ \text{Average Cross-Entropy Loss} = \frac{1}{n} \sum_{i=1}^{n} - P(y_i = 0 | \textbf{X}_i) \ln \hat{P_\theta}(y_i = 0 | \textbf{X}_i) - P(y_i = 1 | \textbf{X}_i) \ln \hat{P_\theta}(y_i = 1 | \textbf{X}_i)$$
$$ \hat{\boldsymbol{\theta}} = \displaystyle\arg \min_{\substack{\theta}} (\text{Average Cross-Entropy Loss}) $$

Since the label $y_i$ is a known value, the probability that $y_i = 1$, $P(y_i = 1 | \textbf{X}_i)$, is equal to $y_i$ and $P(y_i = 0 | \textbf{X}_i)$ is equal to $1 - y_i$. The model's probability distribution $\hat{P_\boldsymbol{\theta}}$ is given by the output of the sigmoid function discussed in the previous two sections. After making these substitutions, we arrive at the average cross-entropy loss equation:

$$ \text{Average Cross-Entropy Loss} = \frac{1}{n} \sum_i \left(- y_i \ln (f_\hat{\boldsymbol{\theta}}(\textbf{X}_i)) - (1 - y_i) \ln (1 - f_\hat{\boldsymbol{\theta}}(\textbf{X}_i) \right) $$
$$ \hat{\boldsymbol{\theta}} = \displaystyle\arg \min_{\substack{\theta}} (\text{Average Cross-Entropy Loss}) $$



## Statistical justification for Cross-Entropy Loss

The cross-entropy loss also has fundamental underpinnings in statistics. Since the logistic regression model predicts probabilities, given a particular logistic model we can ask, "What is the probability that this model produced the set of observed outcomes $ \textbf{y} $?" We might naturally adjust the parameters of our model until the probability of drawing our dataset from the model is as high as possible. Although we will not prove it in this section, this procedure is equivalent to minimizing the cross-entropy loss—this is the *maximum likelihood* statistical justification for the cross-entropy loss.

## Summary

Average KL divergence can be interpreted as the average log difference between the two distributions $P$ and $\hat{P_\boldsymbol{\theta}}$ weighted by $P$. Minimizing average KL divergence also minimizes average cross-entropy loss. We can reduce the divergence of logistic regression models by selecting parameters that accurately classify commonly occurring data.
