
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Predicting-Tip-Amounts" data-toc-modified-id="Predicting-Tip-Amounts-1">Predicting Tip Amounts</a></span></li><li><span><a href="#Defining-a-Linear-Model" data-toc-modified-id="Defining-a-Linear-Model-2">Defining a Linear Model</a></span><ul class="toc-item"><li><span><a href="#Estimating-the-Linear-Model" data-toc-modified-id="Estimating-the-Linear-Model-2.1">Estimating the Linear Model</a></span></li></ul></li></ul></div>


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

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
```

## Predicting Tip Amounts

Previously, we worked with a dataset that contained one row for each table that a waiter served in a week. Our waiter collected this data in order to predict the tip amount he could expect to receive from a future table.


```python
tips = sns.load_dataset('tips')
tips.head()
```


```python
sns.distplot(tips['tip'], bins=25);
```

As we have covered previously, if we choose a constant model and the mean squared error cost, our model will predict the mean of the tip amount:


```python
np.mean(tips['tip'])
```

This means that if a new party orders a meal and the waiter asks us how much tip he will likely receive, we will say "around \$3", no matter how large the table is or how much their total bill was.

However, looking at other variables in the dataset suggest that we might be able to make more accurate predictions if we incorporate them into our model. For example, the following plot of the tip amount against the total bill shows a positive association.


```python
# HIDDEN
sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False)
plt.title('Tip amount vs. Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount');
```

Although the average tip amount is \$3, if a table orders \$40 worth of food we would certainly expect that the waiter receives more than \$3 of tip. Thus, we would like to alter our model so that it makes predictions based on the variables in our dataset instead of blindly predicting the mean tip amount. To do this, we use a linear model instead of constant one.

Let's briefly review our current toolbox for modeling and estimation and define some new notation so that we can better represent the additional complexity that linear models have.

## Defining a Simple Linear Model

We are interested in predicting the tip amount based on the total bill of a table. We use $ y $ to represent the tip amount, the variable we are trying to predict. We use $ x $ to represent the total bill, the variable we use for prediction. 

We define a linear model $ f_\theta $ that depends on $ x $:

$$
f_\theta (x) = \theta_1 x + \theta_0
$$

and treat it as the true generating function of the data.

$ f_\theta (x) $ assumes that in truth, $ y $ has a perfectly linear relationship with $ x $. However, our observed data do not follow a perfectly straight line because of some random noise $ \epsilon $. Mathematically, we account for this by adding a noise term:

$$
y = f_\theta (x) + \epsilon
$$

If the assumption that $ y $ has a perfectly linear relationship with $ x $ holds, and we are able to somehow find the exact values of $ \theta_1 $ and $ \theta_0 $, and we magically have no random noise, we will be able to perfectly predict the amount of tip the waiter will get for all tables, forever. Of course, we cannot completely fulfill any of these criteria in practice. Instead, we will estimate $ \theta_1 $ and $ \theta_0 $ using our dataset to make our predictions as accurate as possible.

### Estimating the Linear Model

Since we cannot find $ \theta_1 $ and $ \theta_0 $ exactly, we will assume that our dataset approximates our population and use our dataset to estimate these parameters. We denote our estimations with $ \hat{\theta_1} $ and $ \hat{\theta_0} $ and define our model as:

$$
f_\hat{\theta} (x) = \hat{\theta_1} x + \hat{\theta_0}
$$

Sometimes you will see $ h(x) $ written instead of $ f_\hat{\theta} (x) $; the "$ h $" stands for hypothesis, as $ f_\hat{\theta} (x) $ is our hypothesis of $ f_{\theta} (x) $.

In order to determine $ \hat{\theta_1} $ and $ \hat{\theta_0} $, we choose a cost function and minimize it using gradient descent.
