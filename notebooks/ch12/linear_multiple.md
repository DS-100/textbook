
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Multiple-Linear-Regression" data-toc-modified-id="Multiple-Linear-Regression-1">Multiple Linear Regression</a></span></li><li><span><a href="#MSE-Cost-and-its-Gradient" data-toc-modified-id="MSE-Cost-and-its-Gradient-2">MSE Cost and its Gradient</a></span></li><li><span><a href="#Fitting-the-Model-With-Gradient-Descent" data-toc-modified-id="Fitting-the-Model-With-Gradient-Descent-3">Fitting the Model With Gradient Descent</a></span></li><li><span><a href="#Visualizing-our-Predictions" data-toc-modified-id="Visualizing-our-Predictions-4">Visualizing our Predictions</a></span></li><li><span><a href="#Using-All-the-Data" data-toc-modified-id="Using-All-the-Data-5">Using All the Data</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-6">Summary</a></span></li></ul></div>


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
pd.options.display.max_columns = 7
np.set_printoptions(precision=2, suppress=True)
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
from scipy.optimize import minimize as sci_min
def minimize(cost_fn, grad_cost_fn, X, y, progress=True):
    '''
    Uses scipy.minimize to minimize cost_fn using a form of gradient descent.
    '''
    theta = np.zeros(X.shape[1])
    iters = 0
    
    def objective(theta):
        return cost_fn(theta, X, y)
    def gradient(theta):
        return grad_cost_fn(theta, X, y)
    def print_theta(theta):
        nonlocal iters
        if progress and iters % progress == 0:
            print(f'theta: {theta} | cost: {cost_fn(theta, X, y):.2f}')
        iters += 1
        
    print_theta(theta)
    return sci_min(
        objective, theta, method='BFGS', jac=gradient, callback=print_theta,
        tol=1e-7
    ).x
```

## Multiple Linear Regression

Our simple linear model has a key advantage over the constant model: it uses the data when making predictions. However, it is still rather limited since it can only use one variable in our dataset when many datasets have many potentially useful variables. For example, consider the following dataset on car models and their milage per gallon (MPG):


```python
mpg = pd.read_csv('mpg.csv').dropna().reset_index(drop=True)
mpg
```

It seems likely that multiple attributes of a car model will affect its MPG. For example, the MPG seems to decrease as horsepower increases:


```python
# HIDDEN
sns.lmplot(x='horsepower', y='mpg', data=mpg);
```

However, cars released later generally have better MPG than older cars:


```python
sns.lmplot(x='model year', y='mpg', data=mpg);
```

It seems possible that we can get a more accurate model if we could take both horsepower and model year into account when making predictions about the MPG. In fact, perhaps the best model takes into account all the numerical variables in our dataset. We can extend our univariate linear regression to allow prediction based on any number of attributes.

We state the following model:

$$
f_\hat{\theta} (x) = \hat{\theta_0} + \hat{\theta_1} x_1 + \ldots + \hat{\theta_p} x_p
$$

Where $ x $ now represents a vector containing $ p $ attributes of a single car. The model above says, "Take multiple attributes of a car, multiply them by some weights, and add them together to make a prediction for MPG."


For example, if we're making a prediction on the first car in our dataset using the horsepower, weight, and model year columns, the vector $ x $ looks like:


```python
# HIDDEN
mpg.loc[0:0, ['horsepower', 'weight', 'model year']]
```

In there examples, we've kept the column names for clarity but keep in mind that $ x $ only contains the numerical values of the table above: $ x = [130.0, 3504.0, 70] $.

Now, we will perform a notational trick that will greatly simplify later formulas. We will prepend the value $ 1 $ to the vector $ x $, so that we have the following vector for $ x $:


```python
# HIDDEN
mpg_mat = mpg.assign(bias=1)
mpg_mat.loc[0:0, ['bias', 'horsepower', 'weight', 'model year']]
```

Now, observe what happens to the formula for our model:

$$
\begin{aligned}
f_\hat{\theta} (x)
&= \hat{\theta_0} + \hat{\theta_1} x_1 + \ldots + \hat{\theta_p} x_p \\
&= \hat{\theta_0} (1) + \hat{\theta_1} x_1 + \ldots + \hat{\theta_p} x_p \\
&= \hat{\theta_0} x_0 + \hat{\theta_1} x_1 + \ldots + \hat{\theta_p} x_p \\
f_\hat{\theta} (x) &= \hat{\theta} \cdot x
\end{aligned}
$$

Where $ \hat{\theta} \cdot x $ is the vector dot product of $ \hat{\theta} $ and $ x $. Vector and matrix notation was designed to succinctly write linear combinations and is thus well-suited for our linear models. However, you will have to remember from now on that $ \hat{\theta} \cdot x $ is a vector-vector dot product. When in doubt, you can always expand the dot product into simple multiplications and additions.

Now, we define the matrix $ X $ as the matrix containing every car model as a row. For example, here are the first five rows of $ X $:


```python
# HIDDEN
mpg_mat = mpg.assign(bias=1)
mpg_mat.loc[0:4, ['bias', 'horsepower', 'weight', 'model year']]
```

Again, keep in mind that the actual matrix $ X $ only contains the numerical values of the table above.

Notice that $ X $ is composed of multiple $ x $ vectors stacked on top of each other. To keep the notation clear, we define $ X_{i} $ to refer to the row with index $ i $ of $ X $. We define $ X_{ij} $ to refer to the element with index $ j $ of the row with index $ i $ of $ X $. Thus, $ X_i $ is a $ p $-dimensional vector and $ X_{ij} $ is a scalar. $ X $ is an $ n \times p $ matrix.

For example, from the table above we have $ X_4 = [1, 140, 3449, 70] $ and $ X_{41} = 140 $. This notation becomes important when we define the cost function since we will need both $ X $, the matrix of input values, and $ y $, the vector of MPGs.

## MSE Cost and its Gradient

The mean squared error cost function takes in a vector of weights $ \hat{\theta} $, a matrix of inputs $ X $, and a vector of observed MPGs $ y $:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))^2\\
\end{aligned}
$$

We've previously derived the gradient of the MSE cost with respect to $ \hat{\theta} $:

$$
\begin{aligned}
\nabla_{\hat\theta} L(\hat{\theta}, X, y)
&= -\frac{2}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))(\nabla_{\hat\theta} f_\hat{\theta} (X_i))\\
\end{aligned}
$$

We know that:

$$
\begin{aligned}
f_\hat{\theta} (x) &= \hat{\theta} \cdot x \\
\end{aligned}
$$

Let's now compute $ \nabla_{\hat\theta} f_\hat{\theta} (x) $. The result is surprisingly simple because $ \hat{\theta} \cdot x = \hat{\theta_0} x_0 + \ldots + \hat{\theta_p} x_p $ and thus $ \frac{\partial}{\partial \theta_0}(\hat{\theta} \cdot x) = x_0 $, $ \frac{\partial}{\partial \theta_1}(\hat{\theta} \cdot x) = x_1 $, and so on.

$$
\begin{aligned}
\nabla_{\hat\theta} f_\hat{\theta} (x)
&= \nabla_{\hat\theta} [ \hat{\theta} \cdot x ] \\
&= \begin{bmatrix}
     \frac{\partial}{\partial \theta_0} (\hat{\theta} \cdot x) \\
     \frac{\partial}{\partial \theta_1} (\hat{\theta} \cdot x) \\
     \vdots \\
     \frac{\partial}{\partial \theta_p} (\hat{\theta} \cdot x) \\
   \end{bmatrix} \\
&= \begin{bmatrix}
     x_0 \\
     x_1 \\
     \vdots \\
     x_p
   \end{bmatrix} \\
\nabla_{\hat\theta} f_\hat{\theta} (x) &= x
\end{aligned}
$$

Finally, we plug this result back into our gradient calculations:

$$
\begin{aligned}
\nabla_{\hat\theta} L(\hat{\theta}, X, y)
&= -\frac{2}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))(\nabla_{\hat\theta} f_\hat{\theta} (X_i))\\
&= -\frac{2}{n} \sum_{i}(y_i - \hat{\theta} \cdot X_i)(X_i)\\
\end{aligned}
$$

Remember that since $ y_i - \hat{\theta} \cdot X_i $ is a scalar and $ X_i $ is a $ p $-dimensional vector, the gradient $ \nabla_{\hat\theta} L(\hat{\theta}, X, y) $ is a $ p $-dimensional vector.

We saw this same type of result when we computed the gradient for univariate linear regression and found that it was 2-dimensional since $ \theta $ was 2-dimensional.

## Fitting the Model With Gradient Descent

We can now plug in our cost and its derivative into gradient descent. As usual, we will define the model, cost, and gradient cost in Python.


```python
def linear_model(thetas, X):
    '''Returns predictions by a linear model on x_vals.'''
    return X @ thetas

def mse_cost(thetas, X, y):
    return np.mean((y - linear_model(thetas, X)) ** 2)

def grad_mse_cost(thetas, X, y):
    n = len(X)
    return -2 / n * (X.T @ y  - X.T @ X @ thetas)
```


```python
# HIDDEN
thetas = np.array([1, 1, 1, 1])
X = np.array([[2, 1, 0, 1], [1, 2, 3, 4]])
y = np.array([3, 9])
assert np.allclose(linear_model(thetas, X), [4, 10])
assert np.allclose(mse_cost(thetas, X, y), 1.0)
assert np.allclose(grad_mse_cost(thetas, X, y), [ 3.,  3.,  3.,  5.])
assert np.allclose(grad_mse_cost(thetas, X + 1, y), [ 25.,  25.,  25.,  35.])
```

Now, we can simply plug in our functions into our gradient descent minimizer:


```python
# HIDDEN
X = (mpg_mat
     .loc[:, ['bias', 'horsepower', 'weight', 'model year']]
     .as_matrix())
y = mpg_mat['mpg'].as_matrix()
```


```python
%%time 

thetas = minimize(mse_cost, grad_mse_cost, X, y)
print(f'theta: {thetas} | cost: {mse_cost(thetas, X, y):.2f}')
```

## Visualizing our Predictions

How does our model do? We can see that the cost decreased dramatically (from 610 to 11.6). We can show the predictions of our model alongside the original values:


```python
# HIDDEN
reordered = ['predicted_mpg', 'mpg', 'horsepower', 'weight', 'model year']
with_predictions = (
    mpg
    .assign(predicted_mpg=linear_model(thetas, X))
    .loc[:, reordered]
)
with_predictions
```

Since we found $ \hat{\theta} $ from gradient descent, we can verify for the first row of our data that $ \hat{\theta} \cdot X_0 $ matches our prediction above:


```python
print(f'Prediction for first row: '
      f'{thetas[0] + thetas[1] * 130 + thetas[2] * 3504 + thetas[3] * 70:.2f}')
```

We've included a widget below to pan through the predictions and the data used to make the prediction:


```python
# HIDDEN
df_interact(with_predictions)
```

We can also plot the residuals of our predictions (actual values - predicted values):


```python
resid = y - linear_model(thetas, X)
plt.scatter(np.arange(len(resid)), resid, s=15)
plt.title('Residuals (actual MPG - predicted MPG)')
plt.xlabel('Index of row in data')
plt.ylabel('MPG');
```

It looks like our model makes reasonable predictions for many car models, although there are some predictions that were off by over 10 MPG (some cars had under 10 MPG!). Perhaps we are more interested in the percent error between the predicted MPG values and the actual MPG values:


```python
resid_prop = resid / with_predictions['mpg']
plt.scatter(np.arange(len(resid_prop)), resid_prop, s=15)
plt.title('Residual proportions (resid / actual MPG)')
plt.xlabel('Index of row in data')
plt.ylabel('Error proportion');
```

It looks like our model's predictions are usually around 20% away from the actual MPG values.

## Using All the Data

Notice that in our example thus far, our $ X $ matrix has four columns: one column of all ones, the horsepower, the weight, and the model year. However, model allows us to handle an arbitrary number of columns:

$$
\begin{aligned}
f_\hat{\theta} (x) &= \hat{\theta} \cdot x
\end{aligned}
$$

As we include more columns into our data matrix, we extend $ \theta $ so that it has one parameter for each column in $ X $. Instead of only selecting three numerical columns for prediction, why not use all seven of them?


```python
# HIDDEN
cols = ['bias', 'cylinders', 'displacement', 'horsepower',
        'weight', 'acceleration', 'model year', 'origin']
X = mpg_mat[cols].as_matrix()
mpg_mat[cols]
```


```python
%%time 

thetas_all = minimize(mse_cost, grad_mse_cost, X, y, progress=10)
print(f'theta: {thetas_all} | cost: {mse_cost(thetas_all, X, y):.2f}')
```

We see that our cost has decreased from 11.6 with three columns of our dataset to 10.85 when using all seven numerical columns of our dataset. We display the proportion error plots for both old and new predictions below:


```python
# HIDDEN
resid_prop_all = (y - linear_model(thetas_all, X)) / with_predictions['mpg']
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(np.arange(len(resid_prop)), resid_prop, s=15)
plt.title('Residual proportions using 3 columns')
plt.xlabel('Index of row in data')
plt.ylabel('Error proportion')
plt.ylim(-0.7, 0.7)

plt.subplot(122)
plt.scatter(np.arange(len(resid_prop_all)), resid_prop_all, s=15)
plt.title('Residual proportions using 7 columns')
plt.xlabel('Index of row in data')
plt.ylabel('Error proportion')
plt.ylim(-0.7, 0.7)

plt.tight_layout();
```

Although the difference is slight, you can see that the errors are a bit lower when using seven columns compared to using three. Both models are much better than using a constant model, as the below plot shows:


```python
# HIDDEN
constant_resid_prop = (y - with_predictions['mpg'].mean()) / with_predictions['mpg']
plt.scatter(np.arange(len(constant_resid_prop)), constant_resid_prop, s=15)
plt.title('Residual proportions using constant model')
plt.xlabel('Index of row in data')
plt.ylabel('Error proportion')
plt.ylim(-1, 1);
```

Using a constant model results in over 75% error for many car MPGs!

## Summary

We have introduced the linear model for regression. Unlike the constant model, the linear model takes in features of our data into account when making predictions, making it much more useful whenever we have correlations between variables of our data.

The procedure of fitting a model to data should now be quite familiar:

1. Select a model.
1. Select a cost function.
1. Minimize the cost function using gradient descent.

It is useful to know that we can usually tweak one of the components without changing the others. In this section, we introduced the linear model without changing our cost function or using a different minimization algorithm. Although modeling can get complicated, it is usually easier to learn by focusing on one component at a time, then combining different parts together as needed in practice.
