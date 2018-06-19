
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#L2-Regularization:-Ridge-Regression" data-toc-modified-id="L2-Regularization:-Ridge-Regression-1">L2 Regularization: Ridge Regression</a></span></li><li><span><a href="#L2-Regularization-Definition" data-toc-modified-id="L2-Regularization-Definition-2">L2 Regularization Definition</a></span><ul class="toc-item"><li><span><a href="#The-Regularization-Parameter" data-toc-modified-id="The-Regularization-Parameter-2.1">The Regularization Parameter</a></span></li><li><span><a href="#Bias-Term-Exclusion" data-toc-modified-id="Bias-Term-Exclusion-2.2">Bias Term Exclusion</a></span></li><li><span><a href="#Data-Normalization" data-toc-modified-id="Data-Normalization-2.3">Data Normalization</a></span></li></ul></li><li><span><a href="#Using-Ridge-Regression" data-toc-modified-id="Using-Ridge-Regression-3">Using Ridge Regression</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4">Summary</a></span></li></ul></div>


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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 5
pd.set_option('precision', 2)
pd.set_option('display.float_format', '{:.2f}'.format)
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
df = pd.read_csv('water_large.csv')
```


```python
# HIDDEN
from collections import namedtuple
Curve = namedtuple('Curve', ['xs', 'ys'])

def flatten(seq): return [item for subseq in seq for item in subseq]

def make_curve(clf, x_start=-50, x_end=50):
    xs = np.linspace(x_start, x_end, num=100)
    ys = clf.predict(xs.reshape(-1, 1))
    return Curve(xs, ys)

def plot_data(df=df, ax=plt, **kwargs):
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=50, **kwargs)

def plot_curve(curve, ax=plt, **kwargs):
    ax.plot(curve.xs, curve.ys, **kwargs)
    
def plot_curves(curves, cols=2, labels=None):
    if labels is None:
        labels = [f'Deg {deg} poly' for deg in degrees]
    rows = int(np.ceil(len(curves) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8),
                             sharex=True, sharey=True)
    for ax, curve, label in zip(flatten(axes), curves, labels):
        plot_data(ax=ax, label='Training data')
        plot_curve(curve, ax=ax, label=label)
        ax.set_ylim(-5e10, 170e10)
        ax.legend()
        
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off',
                    left='off', right='off')
    plt.grid(False)
    plt.title('Polynomial Regression')
    plt.xlabel('Water Level Change (m)')
    plt.ylabel('Water Flow (Liters)')
    plt.tight_layout()
    
```


```python
# HIDDEN
def coefs(clf):
    reg = clf.named_steps['reg']
    return np.append(reg.intercept_, reg.coef_)

def coef_table(clf):
    vals = coefs(clf)
    return (pd.DataFrame({'Coefficient Value': vals})
            .rename_axis('degree'))
```


```python
# HIDDEN
X = df.iloc[:, [0]].as_matrix()
y = df.iloc[:, 1].as_matrix()

degrees = [1, 2, 8, 12]
clfs = [Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                  ('reg', LinearRegression())])
        .fit(X, y)
        for deg in degrees]

curves = [make_curve(clf) for clf in clfs]

alphas = [0.01, 0.1, 1.0, 10.0]

ridge_clfs = [Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                        ('reg', RidgeCV(alphas=alphas, normalize=True))])
        .fit(X, y)
        for deg in degrees]

ridge_curves = [make_curve(clf) for clf in ridge_clfs]
```

## L2 Regularization: Ridge Regression

In this section we introduce $ L_2 $ regularization, a method of penalizing large weights in our cost function to lower model variance. We briefly review linear regression, then introduce regularization as a modification to the cost function.

To perform least squares linear regression, we use the model:

$$
f_\hat{\theta}(x) = \hat{\theta} \cdot x
$$

We fit the model by minimizing the mean squared error cost function:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i}^n(y_i - f_\hat{\theta} (X_i))^2\\
\end{aligned}
$$

In the above definitions, $ X $ represents the $ n \times p $ data matrix, $ x $ represents a row of $ X $, $ y $ represents the observed outcomes, and $ \hat{\theta} $ represents the model weights.

## L2 Regularization Definition

To add $ L_2 $ regularization to the model, we modify the cost function above:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))^2
    + \lambda \sum_{j = 1}^{p} \hat{\theta_j}^2
\end{aligned}
$$

Notice that the cost function above is the same as before with the addition of the $ L_2 $ regularization $ \lambda \sum_{j = 1}^{p} \hat{\theta_j}^2 $ term. The summation in this term sums the square of each model weight $ \hat{\theta_1}, \hat{\theta_2}, \ldots, \hat{\theta_p} $. The term also introduces a new scalar model parameter $ \lambda $ that adjusts the regularization penalty.

The regularization term causes the cost to increase if the values in $ \hat{\theta} $ are further away from 0. With the addition of regularization, the optimal model weights minimize the combination of loss and regularization penalty rather than the loss alone. Since the resulting model weights tend to be smaller in absolute value, the model has lower variance and higher bias.

Using $ L_2 $ regularization with a linear model and the mean squared error cost function is also known more commonly as **ridge regression**.

### The Regularization Parameter

The regularization parameter $ \lambda $ controls the regularization penalty. A small $ \lambda $ results in a small penalty—if $ \lambda = 0 $ the regularization term is also $ 0 $ and the cost is not regularized at all.

A large $ \lambda $ terms results in a large penalty and therefore a simpler model. Increasing $ \lambda $ decreases the variance and increases the bias of the model. We use cross-validation to select the value of $ \lambda $ that minimizes the validation error.

**Note about regularization in `scikit-learn`:**

`scikit-learn` provides regression models that have regularization built-in. For example, to conduct ridge regression you may use the [`sklearn.linear_model.Ridge`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) regression model. Note that `scikit-learn` models call the regularization parameter `alpha` instead of $ \lambda $.

`scikit-learn` conveniently provides regularized models that perform cross-validation to select a good value of $ \lambda $. For example, the [`sklearn.linear_model.RidgeCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV) allows users to input regularization parameter values and will automatically use cross-validation to select the parameter value with the least validation error.

### Bias Term Exclusion

Note that the bias term $ \theta_0 $ is not included in the summation of the regularization term. We do not penalize the bias term because increasing the bias term does not increase the variance of our model—the bias term simply shifts all predictions by a constant value.

### Data Normalization

Notice that the regularization term $ \lambda \sum_{j = 1}^{p} \hat{\theta_j}^2 $ penalizes each $ \hat{\theta_j} $ equally. However, the effect of each $ \hat{\theta_j} $ differs depending on the data itself. Consider this section of the water flow dataset after adding degree 8 polynomial features:


```python
# HIDDEN
pd.DataFrame(clfs[2].named_steps['poly'].transform(X[:5]),
             columns=[f'deg_{n}_feat' for n in range(8)])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deg_0_feat</th>
      <th>deg_1_feat</th>
      <th>...</th>
      <th>deg_6_feat</th>
      <th>deg_7_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-15.94</td>
      <td>253.98</td>
      <td>...</td>
      <td>-261095791.08</td>
      <td>4161020472.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-29.15</td>
      <td>849.90</td>
      <td>...</td>
      <td>-17897014961.65</td>
      <td>521751305227.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.19</td>
      <td>1309.68</td>
      <td>...</td>
      <td>81298431147.09</td>
      <td>2942153527269.12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37.49</td>
      <td>1405.66</td>
      <td>...</td>
      <td>104132296999.30</td>
      <td>3904147586408.71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-48.06</td>
      <td>2309.65</td>
      <td>...</td>
      <td>-592123531634.12</td>
      <td>28456763821657.78</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 8 columns</p>
</div>



We can see that the degree 7 polynomial features have much larger values than the degree 1 features. This means that a large model weight for the degree 7 features affects the predictions much more than a large model weight for the degree 1 features. If we apply regularization to this data directly, the regularization penalty will disproportionately lower the model weight for the lower degree features. In practice, this often results in high model variance even after applying regularization since the features with large effect on prediction will not be affected.

To combat this, we *normalize* each data column by subtracting the mean and scaling the values in each column to be between -1 and 1. In `scikit-learn`, most regression models allow initializing with `normalize=True` to normalize the data before fitting.

Another analogous technique is *standardizing* the data columns by subtracting the mean and dividing by the standard deviation for each data column.

## Using Ridge Regression

We have previously used polynomial features to fit polynomials of degree 2, 8, and 12 to water flow data. The original data and resulting model predictions are repeated below.


```python
# HIDDEN
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>water_level_change</th>
      <th>water_flow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-15.94</td>
      <td>60422330445.52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-29.15</td>
      <td>33214896575.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.19</td>
      <td>972706380901.06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7.09</td>
      <td>236352046523.78</td>
    </tr>
    <tr>
      <th>21</th>
      <td>46.28</td>
      <td>1494256381086.73</td>
    </tr>
    <tr>
      <th>22</th>
      <td>14.61</td>
      <td>378146284247.97</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 2 columns</p>
</div>




```python
# HIDDEN
plot_curves(curves)
```


![png](reg_ridge_files/reg_ridge_16_0.png)


To conduct ridge regression, we first extract the data matrix and the vector of outcomes from the data:


```python
X = df.iloc[:, [0]].as_matrix()
y = df.iloc[:, 1].as_matrix()
print('X: ')
print(X)
print()
print('y: ')
print(y)
```

    X: 
    [[-15.94]
     [-29.15]
     [ 36.19]
     ...
     [  7.09]
     [ 46.28]
     [ 14.61]]
    
    y: 
    [6.04e+10 3.32e+10 9.73e+11 ... 2.36e+11 1.49e+12 3.78e+11]


Then, we apply a degree 12 polynomial transform to `X`:


```python
from sklearn.preprocessing import PolynomialFeatures

# We need to specify include_bias=False since sklearn's classifiers
# automatically add the intercept term.
X_poly_8 = PolynomialFeatures(degree=8, include_bias=False).fit_transform(X)
print('First two rows of transformed X:')
print(X_poly_8[0:2])
```

    First two rows of transformed X:
    [[-1.59e+01  2.54e+02 -4.05e+03  6.45e+04 -1.03e+06  1.64e+07 -2.61e+08
       4.16e+09]
     [-2.92e+01  8.50e+02 -2.48e+04  7.22e+05 -2.11e+07  6.14e+08 -1.79e+10
       5.22e+11]]


We specify `alpha` values that `scikit-learn` will select from using cross-validation, then use the `RidgeCV` classifier to fit the transformed data.


```python
from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1.0, 10.0]

# Remember to set normalize=True to normalize data
clf = RidgeCV(alphas=alphas, normalize=True).fit(X_poly_8, y)

# Display the chosen alpha value:
clf.alpha_
```




    0.1



Finally, we plot the model predictions for the base degree 8 polynomial classifier next to the regularized degree 8 classifier:


```python
# HIDDEN
fig = plt.figure(figsize=(10, 5))

plt.subplot(121)
plot_data()
plot_curve(curves[2])
plt.title('Base degree 8 polynomial')

plt.subplot(122)
plot_data()
plot_curve(ridge_curves[2])
plt.title('Regularized degree 8 polynomial')
plt.tight_layout()
```


![png](reg_ridge_files/reg_ridge_24_0.png)


We can see that the regularized polynomial is smoother than the base degree 8 polynomial and still captures the major trend in the data.

Comparing the coefficients of the non-regularized and regularized models shows that ridge regression favors placing model weights on the lower degree polynomial terms:


```python
# HIDDEN
base = coef_table(clfs[2]).rename(columns={'Coefficient Value': 'Base'})
ridge = coef_table(ridge_clfs[2]).rename(columns={'Coefficient Value': 'Regularized'})

pd.options.display.max_rows = 20
display(base.join(ridge))
pd.options.display.max_rows = 7
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Base</th>
      <th>Regularized</th>
    </tr>
    <tr>
      <th>degree</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>225782472111.94</td>
      <td>221063525725.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13115217770.78</td>
      <td>6846139065.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-144725749.98</td>
      <td>146158037.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-10355082.91</td>
      <td>1930090.04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>567935.23</td>
      <td>38240.62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9805.14</td>
      <td>564.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-249.64</td>
      <td>7.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.09</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.03</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


Repeating the process for degree 12 polynomial features results in a similar result:


```python
# HIDDEN
fig = plt.figure(figsize=(10, 5))

plt.subplot(121)
plot_data()
plot_curve(curves[3])
plt.title('Base degree 12 polynomial')
plt.ylim(-5e10, 170e10)

plt.subplot(122)
plot_data()
plot_curve(ridge_curves[3])
plt.title('Regularized degree 12 polynomial')
plt.ylim(-5e10, 170e10)
plt.tight_layout()
```


![png](reg_ridge_files/reg_ridge_28_0.png)


Increasing the regularization parameter results in progressively simpler models. The plot below demonstrates the effects of increasing the regularization amount from 0.001 to 100.0.


```python
# HIDDEN
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

alpha_clfs = [Pipeline([
    ('poly', PolynomialFeatures(degree=12, include_bias=False)),
    ('reg', Ridge(alpha=alpha, normalize=True))]
).fit(X, y) for alpha in alphas]

alpha_curves = [make_curve(clf) for clf in alpha_clfs]
labels = [f'$\\lambda = {alpha}$' for alpha in alphas]

plot_curves(alpha_curves, cols=3, labels=labels)
```


![png](reg_ridge_files/reg_ridge_30_0.png)


As we can see, increasing the regularization parameter increases the bias of our model. If our parameter is too large, the model becomes a constant model because any non-zero model weight is heavily penalized.

## Summary

Using $ L_2 $ regularization allows us to tune model bias and variance by penalizing large model weights. $ L_2 $ regularization for least squares linear regression is also known by the more common name ridge regression. Using regularization adds an additional model parameter $ \lambda $ that we adjust using cross-validation.
