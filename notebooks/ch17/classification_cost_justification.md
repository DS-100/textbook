

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

Since logistic regression outputs probabilities, a logistic model produces a certain type of probability distribution. Specifically, it uses its parameters $ \hat \theta $ to estimate the probability that the label $ y $ is $ 1 $ for an example input $ x $.

For example, suppose that $ x $ is a scalar recording the forecasted chance of rain for the day and $ y = 1 $ means that Mr. Doe takes his umbrella with him to work. A logistic model predicts the probability that Mr. Doe takes his umbrella given a forecasted chance of rain: $ \hat{P_\theta}(y = 1 | x) $.

Collecting data on Mr. Doe's umbrella usage provides us with a method of constructing an empirical probability distribution $ P(y = 1 | x) $. For example, if there were five days where the chance of rain $ x = 0.60 $ and Mr. Doe only took his umbrella to work once, $ P(y = 1 | x = 0.60) = 0.20 $. We can compute a similar probability distribution for each value of $ x $ that appears in our data. Naturally, after fitting a logistic model we would like the distribution predicted by the model to be as close as possible to the empirical distribution from the dataset. That is, for all values of $ x $ that appear in our data, we want:

$$ \hat{P_\theta}(y = 1 | x) \approx P(y = 1 | x) $$

One commonly used metric to determine the "closeness" of two probability distributions is the Kullback–Leibler divergence, or KL divergence, which has its roots in information theory.

## Defining Average KL Divergence

KL divergence quantifies the difference between the probability distribution computed by our logistic model and the actual distribution based on the dataset. Intuitively, it calculates how imprecisely the logistic model estimates the distribution of labels in data.

The KL divergence for binary classification between two distributions $P$ and $\hat{P_\theta}$ for a single data point $(x, y)$ is given by:

$$D(P || \hat{P_\theta}) = P(y = 0 | x) \ln \left(\frac{P(y = 0 | x)}{\hat{P_\theta}(y = 0 | x)}\right) + P(y = 1 | x) \ln \left(\frac{P(y = 1 | x)}{\hat{P_\theta}(y = 1 | x)}\right)$$

KL divergence is not symmetric: $$D(P || \hat{P_\theta}) \neq D(\hat{P_\theta} || P)$$ In other words, the divergence of $\hat{P_\theta}$ from $P$ is not the same as the divergence of $P$ from $\hat{P_\theta}$. Since we use $\hat{P_\theta}$ to approximate $P$, we are concerned with $ D(P || \hat{P_\theta}) $.

The best $\hat{\theta}$ minimizes the average KL divergence of the entire dataset of $n$ points:

$$\displaystyle\arg \min_{\substack{\theta}} \frac{1}{n} \sum_{i=1}^{n} \left(P(y_i = 0 | x_i) \ln \left(\frac{P(y_i = 0 | x_i)}{\hat{P_\theta}(y_i = 0 | x_i)}\right) + P(y_i = 1 | x_i) \ln \left(\frac{P(y_i = 1 | x_i)}{\hat{P_\theta}(y_i = 1 | x_i)}\right)\right)$$

KL divergence does not penalize mismatch for rare events with respect to $P$. If the model predicts a high probability for an event that is actually rare, then both $P(k)$ and $\ln \left(\frac{P(k)}{\hat{P}(k)}\right)$ are low so the divergence is also low. However, if the model predicts a low probability for an event that is actually common, then the divergence is high. We can deduce that a logistic model that accurately predicts common events has a lower divergence from $P$ than does a model that accurately predicts rare events but varies widely on common events. 

## Deriving Cross-Entropy Loss from KL Divergence

The structure of the above average KL divergence equation contains some surface similarities with cross-entropy loss. We will now show with some algebraic manipulation that minimizing average KL divergence is in fact equivalent to minimizing average cross-entropy loss.

Using properties of logarithms, we can rewrite the weighted log ratio:
$$P(y_i = k | x_i) \ln \left(\frac{P(y_i = k | x_i)}{\hat{P_\theta}(y_i = k | x_i)}\right) = P(y_i = k | x_i) \ln P(y_i = k | x_i) - P(y_i = k | x_i) \ln \hat{P_\theta}(y_i = k | x_i)$$

Note that since the first term doesn't depend on $\theta$, it doesn't affect $\displaystyle\arg \min_{\substack{\theta}}$ and can be removed from the equation. The resulting equation is the cross-entropy loss of the model $\hat{P_\theta}$:

$$\displaystyle\arg \min_{\substack{\theta}} \frac{1}{n} \sum_{i=1}^{n} - P(y_i = 0 | x_i) \ln \hat{P_\theta}(y_i = 0 | x_i) - P(y_i = 1 | x_i) \ln \hat{P_\theta}(y_i = 1 | x_i)$$


Since the label $y_i$ is a known value, the probability that $y_i = 1$, $P(y_i = 1 | x_i)$, is equal to $y_i$ and $P(y_i = 0 | x_i)$ is equal to $1 - y_i$. The model's probability distribution $\hat{P_\theta}$ is given by the output of the sigmoid function discussed in the previous two sections. After making these substitutions, we arrive at the cross-entropy loss equation:

$$ \displaystyle\arg \min_{\substack{\theta}} \frac{1}{n} \sum_i \left(- y_i \ln (f_\hat{\theta}(x_i)) - (1 - y_i) \ln (1 - f_\hat{\theta}(x_i) \right) $$

## Statistical justification for Cross-Entropy Loss

The cross-entropy loss also has fundamental underpinnings in statistics. Since the logistic regression model predicts probabilities, given a particular logistic model we can ask, "What is the probability that this model produced the set of observed outcomes $ y $?" We might naturally adjust the parameters of our model until the probability of drawing our dataset from the model is as high as possible. Although we will not prove it in this section, this procedure is equivalent to minimizing the cross-entropy loss—this is the *maximum likelihood* statistical justification for the cross-entropy loss.

## Summary

Average KL divergence can be interpreted as the average log difference between the two distributions $P$ and $\hat{P_\theta}$ weighted by $P$. Minimizing average KL divergence also minimizes average cross-entropy loss. We can reduce the divergence of logistic regression models by selecting parameters that accurately classify commonly occurring data.
