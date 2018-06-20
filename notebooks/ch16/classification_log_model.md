
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#The-Logistic-Model" data-toc-modified-id="The-Logistic-Model-1">The Logistic Model</a></span></li><li><span><a href="#Real-Numbers-to-Probabilities" data-toc-modified-id="Real-Numbers-to-Probabilities-2">Real Numbers to Probabilities</a></span></li><li><span><a href="#The-Logistic-Function" data-toc-modified-id="The-Logistic-Function-3">The Logistic Function</a></span></li><li><span><a href="#Logistic-Model-Definition" data-toc-modified-id="Logistic-Model-Definition-4">Logistic Model Definition</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-5">Summary</a></span></li></ul></div>


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
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```


```python
# HIDDEN
def df_interact(df, nrows=7, ncols=7):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]
    if len(df.columns) <= ncols:
        interact(peek, row=(0, len(df) - nrows, nrows), col=fixed(0))
    else:
        interact(peek,
                 row=(0, len(df) - nrows, nrows),
                 col=(0, len(df.columns) - ncols))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))
```


```python
# HIDDEN
def jitter_df(df, x_col, y_col):
    x_jittered = df[x_col] + np.random.normal(scale=0, size=len(df))
    y_jittered = df[y_col] + np.random.normal(scale=0.05, size=len(df))
    return df.assign(**{x_col: x_jittered, y_col: y_jittered})
```


```python
# HIDDEN
lebron = pd.read_csv('lebron.csv')
```

## The Logistic Model

In this section, we introduce the **logistic model**, a regression model that we use to predict probabilities.

Recall that fitting a model for prediction requires three components: a model that makes predictions, a cost function, and an optimization method. For the by-now familiar least squares linear regression, we select the model:

$$
\begin{aligned}
f_\hat{\theta} (x) &= \hat{\theta} \cdot x
\end{aligned}
$$

And the cost function:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))^2\\
\end{aligned}
$$

We use gradient descent as our optimization method. In the definitions above, $ X $ represents the $ n \times p $ data matrix, $ x $ represents a row of $ X $, $ y $ represents the observed outcomes, and $ \hat{\theta} $ represents the model weights.

## Real Numbers to Probabilities

Observe that the model $ f_\hat{\theta} (x) = \hat{\theta} \cdot x $ can output any real number $ \mathbb{R} $ since it produces a linear combination of the values in the vector $ x $ which itself can contain any value from $ \mathbb{R} $.

We can easily visualize this when $ x $ is a scalar. If $ \hat \theta = 0.5$, our model becomes $ f_\hat{\theta} (x) = 0.5 x $. Its predictions can take on any value from negative infinity to positive infinity:


```python
# HIDDEN
xs = np.linspace(-100, 100, 100)
ys = 0.5 * xs
plt.plot(xs, ys)
plt.xlabel('$x$')
plt.ylabel(r'$f_\hat{\theta}(x)$')
plt.title(r'Model predictions for $ \hat{\theta} = 0.5 $');
```


![png](classification_log_model_files/classification_log_model_7_0.png)


For classification tasks, we want to constrain $ f_\hat{\theta}(x) $ so that its output can be interpreted as a probability. This means that it may only output values in the range $ [0, 1] $. In addition, we would like large values of $ f_\hat{\theta}(x) $ to correspond to high probabilities and small values to low probabilities.

## The Logistic Function

To accomplish this, we introduce the **logistic function**, often called the **sigmoid function**:

$$
\begin{aligned}
\sigma(t) = \frac{1}{1 + e^{-t}}
\end{aligned}
$$

For ease of reading, we often replace $ e^x $ with $ \text{exp}(x) $ and write:

$$
\begin{aligned}
\sigma (t) = \frac{1}{1 + \text{exp}(-t)}
\end{aligned}
$$

We plot the sigmoid function for values of $ t \in [-10, 10] $ below.


```python
# HIDDEN
from scipy.special import expit
xs = np.linspace(-10, 10, 100)
ys = expit(xs)
plt.plot(xs, ys)
plt.title(r'Sigmoid function')
plt.xlabel('$ t $')
plt.ylabel(r'$ \sigma(t) $');
```


![png](classification_log_model_files/classification_log_model_10_0.png)


Observe that the sigmoid function $ \sigma(t) $ takes in any real number $ \mathbb{R} $ and outputs only numbers between 0 and 1. The function is monotonically increasing on its input $ t $; large values of $ t $ correspond to values closer to 1, as desired. This is not a coincidence—the sigmoid function may be derived from a log ratio of probabilities, although we omit the derivation for brevity.

## Logistic Model Definition

We may now take our linear model $ f_\hat{\theta} (x) = \hat{\theta} \cdot x $ and use it as the input to the sigmoid function to create the model:

$$
\begin{aligned}
f_\hat{\theta} (x) = \sigma(\hat{\theta} \cdot x)
\end{aligned}
$$

In other words, we take the output of linear regression—any number in $ \mathbb{R} $— and use the sigmoid function to restrict the model's final output to be a valid probability between zero and one.

This model is called the **logistic model**.

To develop some intuition for how the model behaves, we restrict $ x $ to be a scalar and plot the logistic model's output for several values of $ \hat{\theta} $.


```python
# HIDDEN
def flatten(li): return [item for sub in li for item in sub]

thetas = [-2, -1, -0.5, 2, 1, 0.5]
xs = np.linspace(-10, 10, 100)

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 6))
for ax, theta in zip(flatten(axes), thetas):
    ys = expit(theta * xs)
    ax.plot(xs, ys)
    ax.set_title(r'$ \hat{\theta} = $' + str(theta))

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off',
                left='off', right='off')
plt.grid(False)
plt.xlabel('$x$')
plt.ylabel(r'$ f_\hat{\theta}(x) $')
plt.tight_layout()
```


![png](classification_log_model_files/classification_log_model_14_0.png)


We see that changing $ \hat{\theta} $ changes the sharpness of the curve; the further away from $ 0 $, the sharper the curve.

## Summary

We introduce the logistic model, a new prediction function that outputs probabilities. To construct the model, we use the output of linear regression as the input of the logistic function.
