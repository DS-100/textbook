
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#A-Cost-Function-for-the-Logistic-Model" data-toc-modified-id="A-Cost-Function-for-the-Logistic-Model-1">A Cost Function for the Logistic Model</a></span></li><li><span><a href="#The-Cross-Entropy-Cost" data-toc-modified-id="The-Cross-Entropy-Cost-2">The Cross Entropy Cost</a></span></li><li><span><a href="#Gradient-of-the-Cross-Entropy-Cost" data-toc-modified-id="Gradient-of-the-Cross-Entropy-Cost-3">Gradient of the Cross Entropy Cost</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4">Summary</a></span></li></ul></div>


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
lebron = pd.read_csv('lebron.csv')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-7-4d6d8db4ec60> in <module>()
          1 # HIDDEN
    ----> 2 lebron = pd.read_csv('lebron.csv')
    

    /srv/conda/envs/data100/lib/python3.6/site-packages/pandas/io/parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, escapechar, comment, encoding, dialect, tupleize_cols, error_bad_lines, warn_bad_lines, skipfooter, skip_footer, doublequote, delim_whitespace, as_recarray, compact_ints, use_unsigned, low_memory, buffer_lines, memory_map, float_precision)
        707                     skip_blank_lines=skip_blank_lines)
        708 
    --> 709         return _read(filepath_or_buffer, kwds)
        710 
        711     parser_f.__name__ = name


    /srv/conda/envs/data100/lib/python3.6/site-packages/pandas/io/parsers.py in _read(filepath_or_buffer, kwds)
        447 
        448     # Create the parser.
    --> 449     parser = TextFileReader(filepath_or_buffer, **kwds)
        450 
        451     if chunksize or iterator:


    /srv/conda/envs/data100/lib/python3.6/site-packages/pandas/io/parsers.py in __init__(self, f, engine, **kwds)
        816             self.options['has_index_names'] = kwds['has_index_names']
        817 
    --> 818         self._make_engine(self.engine)
        819 
        820     def close(self):


    /srv/conda/envs/data100/lib/python3.6/site-packages/pandas/io/parsers.py in _make_engine(self, engine)
       1047     def _make_engine(self, engine='c'):
       1048         if engine == 'c':
    -> 1049             self._engine = CParserWrapper(self.f, **self.options)
       1050         else:
       1051             if engine == 'python':


    /srv/conda/envs/data100/lib/python3.6/site-packages/pandas/io/parsers.py in __init__(self, src, **kwds)
       1693         kwds['allow_leading_cols'] = self.index_col is not False
       1694 
    -> 1695         self._reader = parsers.TextReader(src, **kwds)
       1696 
       1697         # XXX


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.__cinit__()


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source()


    FileNotFoundError: File b'lebron.csv' does not exist


## A Cost Function for the Logistic Model

We have defined a regression model for probabilities, the logistic model:

$$
\begin{aligned}
f_\hat{\theta} (x) = \sigma(\hat{\theta} \cdot x)
\end{aligned}
$$

Like the model for linear regression, this model has parameters $ \hat{\theta} $, a vector that contains one parameter for each feature of $ x $. We now address the problem of defining a cost function for this model which allows us to fit the model's parameters to data.

Intuitively, we want the model's predictions to match the data as closely as possible. Below we recreate a plot of LeBron's shot attempts in the 2017 NBA Playoffs using the distance of each shot from the basket. The points are jittered on the y-axis to mitigate overplotting.


```python
# HIDDEN
np.random.seed(42)
sns.lmplot(x='shot_distance', y='shot_made',
           data=lebron,
           fit_reg=False, ci=False,
           y_jitter=0.1,
           scatter_kws={'alpha': 0.3})
plt.title('LeBron Shot Attempts')
plt.xlabel('Distance from Basket (ft)')
plt.ylabel('Shot Made');
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-a6554a34533e> in <module>()
          2 np.random.seed(42)
          3 sns.lmplot(x='shot_distance', y='shot_made',
    ----> 4            data=lebron,
          5            fit_reg=False, ci=False,
          6            y_jitter=0.1,


    NameError: name 'lebron' is not defined


Noticing the large cluster of made shots close to the basket and the smaller cluster of missed shots further from the basket, we expect that a logistic model fitted on this data might look like:


```python
# HIDDEN
from scipy.special import expit

np.random.seed(42)
sns.lmplot(x='shot_distance', y='shot_made',
           data=lebron,
           fit_reg=False, ci=False,
           y_jitter=0.1,
           scatter_kws={'alpha': 0.3})

xs = np.linspace(-2, 32, 100)
ys = expit(-0.15 * (xs - 15))
plt.plot(xs, ys, c='r', label='Logistic model')

plt.title('Possible logistic model fit')
plt.xlabel('Distance from Basket (ft)')
plt.ylabel('Shot Made');
```


![png](classification_cost_files/classification_cost_7_0.png)


Although we can use the mean squared error cost function as we have for linear regression, for a logistic model this cost function is non-convex and thus difficult to optimize.

## The Cross Entropy Cost

Instead of the mean squared error, we use the **cross entropy cost**. Let $ X $ represent the $ n \times p $ input data matrix, $ y $ the vector of observed data values, and $ f_\hat{\theta}(x) $ the logistic model. Using this notation, the cross entropy cost is defined as:

$$
\begin{aligned}
L(\hat{\theta}, X, y) = \frac{1}{n} \sum_i \left(- y_i \ln (f_\hat{\theta}(X_i)) - (1 - y_i) \ln (1 - f_\hat{\theta}(X_i) \right)
\end{aligned}
$$

You may observe that as usual we take the mean loss over each point in our dataset. This loss is called the cross entropy loss and forms the inner expression in the above summation: 

$$
\begin{aligned}
\ell(\hat{\theta}, X, y) = - y_i \ln (f_\hat{\theta}(X_i)) - (1 - y_i) \ln (1 - f_\hat{\theta}(X_i) )
\end{aligned}
$$

Recall that each $ y_i $ is either 0 or 1 in our dataset. If $ y_i = 0 $, the first term in the loss is zero. If $ y_i = 1 $, the second term in the loss is zero. Thus, only one term of the cross entropy loss contributes to the overall cost for each point in our dataset.

Suppose $ y_i = 0 $ and our predicted probability $ f_\hat{\theta}(X_i) = 0 $—our model is completely correct. The loss for this point will be:

$$
\begin{aligned}
\ell(\hat{\theta}, X, y)
&= - y_i \ln (f_\hat{\theta}(X_i)) - (1 - y_i) \ln (1 - f_\hat{\theta}(X_i) ) \\
&= - 0 - (1 - 0) \ln (1 - 0 ) \\
&= - \ln (1) \\
&= 0
\end{aligned}
$$

As expected, the loss for a correct prediction is $ 0 $. You may verify that the further the predicted probability is from the true value, the greater the loss.

Minimizing the overall cross entropy cost requires the model $ f_\hat{\theta}(x) $ to make the most accurate predictions it can. Conveniently, this cost function is convex, making gradient descent a useful choice for optimization.

## Gradient of the Cross Entropy Cost

In order to run gradient descent on the cross entropy cost we must calculate the gradient of the cost function. First, we compute the derivative of the sigmoid function since we'll use it in our gradient calculation.

$$
\begin{aligned}
\sigma(t) &= \frac{1}{1 + e^{-t}} \\
\sigma'(t) &= \frac{e^{-t}}{(1 + e^{-t})^2} \\
\sigma'(t) &= \frac{1}{1 + e^{-t}} \cdot \left(1 - \frac{1}{1 + e^{-t}} \right) \\
\sigma'(t) &= \sigma(t) (1 - \sigma(t))
\end{aligned}
$$

The derivative of the sigmoid function can be conveniently expressed in terms of the sigmoid function itself.

As a shorthand, we define $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $. We will soon need the gradient of $ \sigma_i $ with respect to the vector $ \hat{\theta} $ so we will derive it now using a straightforward application of the chain rule. 

$$
\begin{aligned}
\nabla_{\hat{\theta}} \sigma_i
&= \nabla_{\hat{\theta}} \sigma(X_i \cdot \hat \theta) \\
&= \sigma(X_i \cdot \hat \theta) (1 - \sigma(X_i \cdot \hat \theta))  \nabla_{\hat{\theta}} (X_i \cdot \hat \theta) \\
&= \sigma_i (1 - \sigma_i) X_i 
\end{aligned}
$$

Now, we derive the gradient for the cross entropy cost with respect to the model parameters $ \hat{\theta} $. In the derivation below, we let $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $.

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_i \left(- y_i \ln (f_\hat{\theta}(X_i)) - (1 - y_i) \ln (1 - f_\hat{\theta}(X_i) \right) \\
&= \frac{1}{n} \sum_i \left(- y_i \ln \sigma_i - (1 - y_i) \ln (1 - \sigma_i ) \right) \\
\nabla_{\hat{\theta}} L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_i \left(
    - \frac{y_i}{\sigma_i} \nabla_{\hat{\theta}} \sigma_i
    + \frac{1 - y_i}{1 - \sigma_i} \nabla_{\hat{\theta}} \sigma_i
\right) \\
&= - \frac{1}{n} \sum_i \left(
    \frac{y_i}{\sigma_i} - \frac{1 - y_i}{1 - \sigma_i}
\right) \nabla_{\hat{\theta}} \sigma_i \\
&= - \frac{1}{n} \sum_i \left(
    \frac{y_i}{\sigma_i} - \frac{1 - y_i}{1 - \sigma_i}
\right) \sigma_i (1 - \sigma_i) X_i \\
&= - \frac{1}{n} \sum_i \left(
    y_i - \sigma_i
\right) X_i \\
\end{aligned}
$$

The surprisingly simple gradient expression allows us to fit a logistic model to the cross entropy loss using gradient descent.

## Summary

The cost function for logistic models is based on cross-entropy loss. Since it is a convex function, we use gradient descent to fit the model to the cost. We now have the necessary components of logistic regression: the model, cost function, and minimization procedure.
