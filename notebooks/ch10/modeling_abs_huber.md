
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Absolute-and-Huber-Loss" data-toc-modified-id="Absolute-and-Huber-Loss-1">Absolute and Huber Loss</a></span><ul class="toc-item"><li><span><a href="#Comparing-Mean-Squared-Error-and-Mean-Absolute-Error" data-toc-modified-id="Comparing-Mean-Squared-Error-and-Mean-Absolute-Error-1.1">Comparing Mean Squared Error and Mean Absolute Error</a></span></li><li><span><a href="#Outliers" data-toc-modified-id="Outliers-1.2">Outliers</a></span></li><li><span><a href="#Minimizing-the-Mean-Absolute-Error" data-toc-modified-id="Minimizing-the-Mean-Absolute-Error-1.3">Minimizing the Mean Absolute Error</a></span></li><li><span><a href="#Mean-Squared-Error-and-Mean-Absolute-Error-Comparison" data-toc-modified-id="Mean-Squared-Error-and-Mean-Absolute-Error-Comparison-1.4">Mean Squared Error and Mean Absolute Error Comparison</a></span></li><li><span><a href="#The-Huber-Loss" data-toc-modified-id="The-Huber-Loss-1.5">The Huber Loss</a></span></li></ul></li></ul></div>


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


```python
# HIDDEN
tips = sns.load_dataset('tips')
tips['pcttip'] = tips['tip'] / tips['total_bill'] * 100
```


```python
# HIDDEN
def mse(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def mae(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))
```


```python
# HIDDEN
def compare_mse_mae(thetas, y_vals, xlims, figsize=(10, 7), cols=3):
    if not isinstance(y_vals, np.ndarray):
        y_vals = np.array(y_vals)
    rows = int(np.ceil(len(thetas) / cols))
    plt.figure(figsize=figsize)
    for i, theta in enumerate(thetas):
        ax = plt.subplot(rows, cols, i + 1)
        sns.rugplot(y_vals, height=0.1, ax=ax)
        plt.axvline(theta, linestyle='--',
                    label=rf'$ \hat\theta_0 = {theta} $')
        plt.title(f'MSE = {mse(theta, y_vals):.2f}\n'
                  f'MAE = {mae(theta, y_vals):.2f}')
        plt.xlim(*xlims)
        plt.yticks([])
        plt.legend()
    plt.tight_layout()
```

## Absolute and Huber Loss

Previously, we said that our model is accurate if it minimizes the squared difference between the predictions and the actual values. We used the mean squared error (MSE) to capture this measure of accuracy:

$$
\begin{aligned}
L(\hat{\theta_0}, y)
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \hat{\theta_0})^2\\
\end{aligned}
$$

We used a simple model that always predicts the same number:

$$ \hat{\theta_0} = C $$

Where $ C $ is some constant. When we use this constant model and the MSE function, we found that $ C $ will always be the mean of the data points. When applied to the tips dataset, we found that the constant model should predict $ 16.08\% $ since $ 16.08\% $ is the mean of the tip percents.

Now, we will keep our model the same but switch to a different loss function: the mean absolute error (MAE). Instead taking the squared difference for each point and our prediction, this loss function takes the absolute difference:

$$
\begin{aligned}
L(\hat{\theta_0}, y)
&= \frac{1}{n} \sum_{i = 1}^{n} |y_i - \hat{\theta_0}| \\
\end{aligned}
$$

### Comparing Mean Squared Error and Mean Absolute Error

To get a better sense of how MSE and MAE compare, let's compare their losses on different datasets. First, we'll use our dataset of one point: $ y = [14] $.


```python
# HIDDEN
compare_mse_mae(thetas=[11, 12, 13, 14, 15, 16],
                y_vals=[14], xlims=(10, 17))
```


![png](modeling_abs_huber_files/modeling_abs_huber_7_0.png)


We see that the MSE is usually higher than the MAE since the error is squared. Let's see what happens when have five points: $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $


```python
# HIDDEN
compare_mse_mae(thetas=[12, 13, 14, 15, 16, 17],
                y_vals=[12.1, 12.8, 14.9, 16.3, 17.2],
                xlims=(11, 18))
```


![png](modeling_abs_huber_files/modeling_abs_huber_9_0.png)


Remember that the actual loss values themselves are not very interesting to us; they are only useful for comparing different values of $ \hat{\theta_0} $. Once we choose a loss function, we will look for the $ \hat{\theta_0} $ that produces the least loss. Thus, we are interested in whether the loss functions are minimized by different values of $ \hat{\theta_0} $.

So far, the two loss functions seem to agree on the $ \hat{\theta_0} $ values that produce the least loss for the values that we've tried. If we look a bit closer, however, we will start to see some differences. We first take the losses and plot them against $ \hat{\theta_0} $ for each of the six $ \hat{\theta_0} $ values we tried.


```python
# HIDDEN
thetas = np.array([12, 13, 14, 15, 16, 17])
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mses = [mse(theta, y_vals) for theta in thetas]
maes = [mae(theta, y_vals) for theta in thetas]

plt.scatter(thetas, mses, label='MSE')
plt.scatter(thetas, maes, label='MAE')
plt.title(r'Loss vs. $ \hat{\theta_0} $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \hat{\theta_0} $ Values')
plt.ylabel('Loss')
plt.legend();
```


![png](modeling_abs_huber_files/modeling_abs_huber_11_0.png)


Then, we compute more values of $ \hat{\theta_0} $ so that the curve is smooth:


```python
# HIDDEN
thetas = np.arange(12, 17.1, 0.05)
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mses = [mse(theta, y_vals) for theta in thetas]
maes = [mae(theta, y_vals) for theta in thetas]

plt.plot(thetas, mses, label='MSE')
plt.plot(thetas, maes, label='MAE')
plt.title(r'Loss vs. $ \hat{\theta_0} $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \hat{\theta_0} $ Values')
plt.ylabel('Loss')
plt.legend();
```


![png](modeling_abs_huber_files/modeling_abs_huber_13_0.png)


Then, we zoom into the region between 1.5 and 5 on the y-axis to see the difference in minima more clearly. We've marked the minima with dotted lines.


```python
# HIDDEN
thetas = np.arange(12, 17.1, 0.05)
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mses = [mse(theta, y_vals) for theta in thetas]
maes = [mae(theta, y_vals) for theta in thetas]

plt.plot(thetas, mses, label='MSE')
plt.plot(thetas, maes, label='MAE')
plt.axvline(np.mean(y_vals), c=sns.color_palette()[0], linestyle='--',
            alpha=0.7, label='Minima of MSE')
plt.axvline(np.median(y_vals), c=sns.color_palette()[1], linestyle='--',
            alpha=0.7, label='Minima of MAE')


plt.title(r'Loss vs. $ \hat{\theta_0} $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \hat{\theta_0} $ Values')
plt.ylabel('Loss')
plt.ylim(1.5, 5)
plt.legend();
```


![png](modeling_abs_huber_files/modeling_abs_huber_15_0.png)


We've found empirically that the MSE and MAE can be minimized by different values of $ \hat{\theta_0} $ for the same dataset. However, we don't have a good sense of when they will differ and more importantly, why they differ.

### Outliers

One difference that we can see in the plots of loss vs. $ \hat{\theta_0} $ above lies in the shape of the loss curves. Plotting the MSE results in a parabolic curve resulting from the squared term in the loss function.

Plotting the MAE, on the other hand, results in what looks like a connected series of lines. This makes sense when we consider that the absolute value function is linear, so taking the average of many absolute value functions should produce a semi-linear function.

Since the MSE has a squared error term, it will be more sensitive to outliers. If $ \hat{\theta_0} = 10 $ and a point lies at 110, that point's error term for MSE will be $ (10 - 110)^2 = 10000 $ whereas in the MAE, that point's error term will be $ |10 - 110| = 100 $. We can illustrate this by taking a set of three points $ y = [ 12, 13, 14 ] $ and plotting the loss vs. $ \hat{\theta_0} $ curves for MSE and MAE.

Use the slider below to move the third point further away from the rest of the data and observe what happens to the loss curves. (We've scaled the curves to keep both in view since the MSE has larger values than the MAE.)


```python
# HIDDEN
def compare_mse_mae_curves(y3=14):
    thetas = np.arange(11.5, 26.5, 0.1)
    y_vals = np.array([12, 13, y3])
    
    mses = [mse(theta, y_vals) for theta in thetas]
    maes = [mae(theta, y_vals) for theta in thetas]
    mse_abs_diff = min(mses) - min(maes)
    mses = [loss - mse_abs_diff for loss in mses]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(11.5, 26.5)
    plt.xlabel('Points')
    
    ax = plt.subplot(122)
    plt.plot(thetas, mses, label='MSE')
    plt.plot(thetas, maes, label='MAE')
    plt.xlim(11.5, 26.5)
    plt.ylim(min(maes) - 1, min(maes) + 10)
    plt.xlabel(r'$ \hat{\theta_0} $')
    plt.ylabel('Loss')
    plt.legend()
```


```python
# HIDDEN
interact(compare_mse_mae_curves, y3=(14, 25));
```


<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



We've shown the curves for $ y_3 = 14 $ and $ y_3 = 25 $ below.


```python
# HIDDEN
compare_mse_mae_curves(y3=14)
```


![png](modeling_abs_huber_files/modeling_abs_huber_21_0.png)



```python
# HIDDEN
compare_mse_mae_curves(y3=25)
```


![png](modeling_abs_huber_files/modeling_abs_huber_22_0.png)


As we move the point further away from the rest of the data, the MSE curve moves with it. When $ y_3 = 14 $, both MSE and MAE will be minimized by the same value of $ \hat{\theta_0} $ at $ \hat{\theta_0} = 13 $. However, when $ y_3 = 25 $, the MSE curve will tell us that the best $ \hat{\theta_0} $ is around 16.7 while the MAE will still say that $ \hat{\theta_0} = 13 $ is best.

### Minimizing the Mean Absolute Error

Now that we have a qualitative sense of how the MSE and MAE differ, we can minimize the MAE to make this difference more precise. As before, we will take the derivative of the loss function with respect to $ \hat{\theta_0} $ and set it equal to zero.

This time, however, we have to deal with the fact that the absolute function is not always differentiable. When $ x > 0 $, $ \frac{\partial}{\partial x} |x| = 1 $. When $ x < 0 $, $ \frac{\partial}{\partial x} |x| = -1 $. Although $ |x| $ is not technicaly differentiable at $ x = 0 $, we will set $ \frac{\partial}{\partial x} |x| = 0 $ so that the equations are easier to work with.

Recall that the equation for the MAE is:

$$
\begin{aligned}
L(\hat{\theta_0}, y)
&= \frac{1}{n} \sum_{i = 1}^{n}|y_i - \hat{\theta_0}|\\
&= \frac{1}{n} \left( \sum_{y_i < \hat{\theta_0}}|y_i - \hat{\theta_0}| + \sum_{y_i = \hat{\theta_0}}|y_i - \hat{\theta_0}| + \sum_{y_i > \hat{\theta_0}}|y_i - \hat{\theta_0}| \right)\\
\end{aligned}
$$

In the line above, we've split up the summation into three separate summations: one that has one term for each $ y_i < \hat{\theta_0} $, one for $ y_i = \hat{\theta_0} $, and one for $ y_i > \hat{\theta_0} $. Why make the summation seemingly more complicated? If we know that $ y_i < \hat{\theta_0} $ we also know that $ |y_i - \hat{\theta_0}| < 0 $ and thus $ \frac{\partial}{\partial \hat{\theta_0}} |y_i - \hat{\theta_0}| = -1 $ from before. A similar logic holds for each term above to make taking the derivative much easier.

Now, we take the derivative with respect to $ \hat{\theta_0} $ and set it equal to zero:

$$
\begin{aligned}
\frac{1}{n} \left( \sum_{y_i < \hat{\theta_0}}(-1) + \sum_{y_i = \hat{\theta_0}}(0) + \sum_{y_i > \hat{\theta_0}}(1) \right) &= 0 \\
\sum_{y_i < \hat{\theta_0}}(-1) + \sum_{y_i > \hat{\theta_0}}(1) &= 0 \\
-\sum_{y_i < \hat{\theta_0}}(1) + \sum_{y_i > \hat{\theta_0}}(1) &= 0 \\
\sum_{y_i < \hat{\theta_0}}(1) &= \sum_{y_i > \hat{\theta_0}}(1) \\
\end{aligned}
$$

What does the result above mean? On the left hand side, we have one term for each data point less than $ \hat{\theta_0} $. On the right, we have one for each data point greater than $ \hat{\theta_0} $. Then, in order to satisfy the equation we need to pick a value for $ \hat{\theta_0} $ that has the same number of smaller and larger points. This is the definition for the *median* of a set of numbers. Thus, the minimizing value of $ \hat{\theta_0} $ for the MAE is $ \hat{\theta_0} = \text{median} (y) $.

When we have an odd number of points, the median is simply the middle point when the points are arranged in sorted order. We can see that in the example below with five points, the loss is minimized when $ \hat{\theta_0} $ lies at the median:


```python
# HIDDEN
def points_and_loss(y_vals, xlim, loss_fn=mae):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    maes = [loss_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(*xlim)
    plt.xlabel('Points')
    
    ax = plt.subplot(122)
    plt.plot(thetas, maes)
    plt.xlim(*xlim)
    plt.xlabel(r'$ \hat{\theta_0} $')
    plt.ylabel('Loss')
    plt.legend()
points_and_loss(np.array([10, 11, 12, 14, 15]), (9, 16))
```

    No handles with labels found to put in legend.



![png](modeling_abs_huber_files/modeling_abs_huber_26_1.png)


However, when we have an even number of points, the loss is minimized when $ \hat{\theta_0} $ is any value in between the two central points.


```python
# HIDDEN
points_and_loss(np.array([10, 11, 14, 15]), (9, 16))
```

    No handles with labels found to put in legend.



![png](modeling_abs_huber_files/modeling_abs_huber_28_1.png)


This is not the case when we use the MSE:


```python
# HIDDEN
points_and_loss(np.array([10, 11, 14, 15]), (9, 16), mse)
```

    No handles with labels found to put in legend.



![png](modeling_abs_huber_files/modeling_abs_huber_30_1.png)


### Mean Squared Error and Mean Absolute Error Comparison

Our investigation and the derivation above show that the MSE is easier to differentiate but is more sensitive to outliers than the MAE. The minimizing $ \hat{\theta_0} $ for MSE is the mean of the data points, and the minimizing $ \hat{\theta_0} $ for the MAE is the median of the data points. Notice that the median is robust to outliers while the mean is not! This phenomenon arises from our construction of the two loss functions.

We have also seen that the MSE will be minimized by a unique value of $ \hat{\theta_0} $, whereas the mean absolute value can be minimized by multiple values of $ \hat{\theta_0} $ when there are an even number of data points.

In the examples so far, the ability to differentiate the loss function isn't that useful since we know the exact minimizing value of $ \hat{\theta_0} $ in both cases. However, the ability to differentiate the loss function becomes very important once we start using complicated models. For complicated models, we will not be able to differentiate the loss function by hand and will need a computer to minimize the loss function for us. We will return to this issue when we cover gradient descent and numerical optimization.

### The Huber Loss

A third loss function called the Huber loss combines both the MSE and MAE to create a loss function that is differentiable *and* robust to outliers. The Huber loss accomplishes this by behaving like the MSE function at values close to $ \hat{\theta_0} $ and switching to the absolute loss for values far from $ \hat{\theta_0} $.

As usual, we create a loss function by taking the mean of the Huber losses for each point in our dataset.

Let's see what the Huber loss function outputs for a dataset of $ y = [14] $ as we vary $ \hat{\theta_0} $:


```python
# HIDDEN
def huber_loss(est, y_obs, alpha = 1):
    d = np.abs(est - y_obs)
    return np.where(d < alpha, 
                    (est - y_obs)**2 / 2.0,
                    alpha * (d - alpha / 2.0))

thetas = np.linspace(0, 50, 200)
loss = huber_loss(thetas, np.array([14]), alpha=5)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\hat{\theta_0}$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')
```


![png](modeling_abs_huber_files/modeling_abs_huber_33_0.png)


We can see that the Huber loss is smooth, unlike the MAE. The Huber loss also increases at a linear rate, unlike the quadratic rate of MSE.

The Huber loss does have a drawback, however. Notice that it transitions from the MSE to the MAE once $ \hat{\theta_0} $ gets far enough from the point. We can tweak this "far enough" to get different loss curves. For example, we can make it transition once $ \hat{\theta_0} $ is just one unit away from the observation:


```python
# HIDDEN
loss = huber_loss(thetas, np.array([14]), alpha=1)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\hat{\theta_0}$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')
```


![png](modeling_abs_huber_files/modeling_abs_huber_35_0.png)


Or we can make it transition when $ \hat{\theta_0} $ is ten units away from the observation:


```python
# HIDDEN
loss = huber_loss(thetas, np.array([14]), alpha=10)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\hat{\theta_0}$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')
```


![png](modeling_abs_huber_files/modeling_abs_huber_37_0.png)


This choice results in a different loss curve and can thus result in different values of $ \hat{\theta_0} $. If we want to use the Huber loss function, we have the additional task of setting this transition point to a suitable value.

The Huber loss function is defined mathematically as follows:

$$
L_\alpha(\hat{\theta_0}, y) = \frac{1}{n} \sum_{i=1}^n \begin{cases}
    \frac{1}{2}(y_i - \hat{\theta_0})^2 &  | y_i - \hat{\theta_0} | \le \alpha \\
    \alpha ( |y_i - \hat{\theta_0}| - \frac{1}{2}\alpha ) & \text{otherwise}
\end{cases}
$$

It is more complex than the previous loss functions because it combines both MSE and MAE. The additional parameter $ \alpha $ sets the point where the Huber loss transitions from the MSE to the absolute loss.

Attempting to take the derivative of the Huber loss function is tedious and does not result in an elegant result like the MSE and MAE. Instead, we can use a computational method called gradient descent to find minimizing value of $ \hat{\theta_0} $.
