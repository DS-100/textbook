
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Loss-Functions" data-toc-modified-id="Loss-Functions-1">Loss Functions</a></span><ul class="toc-item"><li><span><a href="#Our-First-Loss-Function:-Mean-Squared-Error" data-toc-modified-id="Our-First-Loss-Function:-Mean-Squared-Error-1.1">Our First Loss Function: Mean Squared Error</a></span></li><li><span><a href="#A-Shorthand" data-toc-modified-id="A-Shorthand-1.2">A Shorthand</a></span></li><li><span><a href="#Finding-the-Best-Value-of-$-\theta-$" data-toc-modified-id="Finding-the-Best-Value-of-$-\theta-$-1.3">Finding the Best Value of $ \theta $</a></span></li><li><span><a href="#The-Minimizing-Value-of-the-Mean-Squared-Error" data-toc-modified-id="The-Minimizing-Value-of-the-Mean-Squared-Error-1.4">The Minimizing Value of the Mean Squared Error</a></span></li><li><span><a href="#Back-to-the-Original-Dataset" data-toc-modified-id="Back-to-the-Original-Dataset-1.5">Back to the Original Dataset</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-1.6">Summary</a></span></li></ul></li></ul></div>


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

## Loss Functions

For now, our model assumes that there is a single population tip percentage $ \theta^* $. We are trying to estimate this parameter, and we use the variable $ \theta $ to denote our estimate. Since our sample of tips is a random sample drawn from the population, we believe that using our sample to create a best estimate $ \hat{\theta} $ will give a value close to $ \theta^* $.

To precisely decide which value of $ \theta $ is best, we need to define a *loss function*. A loss function is a mathematical function that takes in an estimate $ \theta $ and the points in our dataset $y_1, y_2, \ldots, y_n$. It outputs a single number that we can use to choose between two different values of $ \theta $. In mathematical notation, we want to create the function:

$$ L(\theta, y_1, y_2, \ldots, y_n) =\ \ldots $$

By convention, the loss function outputs lower values for preferable values of $ \theta $ and larger values for worse values of $ \theta $. In the previous section, we compared $ \theta = 10 $ and $ \theta = 15 $.


```python
# HIDDEN
sns.distplot(tips['pcttip'], bins=np.arange(30), rug=True)

plt.axvline(x=10, c='darkblue', linestyle='--', label=r'$ \theta = 10$')
plt.axvline(x=15, c='goldenrod', linestyle='--', label=r'$ \theta = 15$')
plt.legend()

plt.xlim(0, 30)
plt.xlabel('Percent Tip Amount')
plt.ylabel('Proportion per Percent');
```


![png](modeling_loss_functions_files/modeling_loss_functions_4_0.png)


Since $ \theta = 15 $ falls closer to most of the points, our loss function should output a small value for $ \theta = 15 $ and a larger value for $ \theta = 10 $.

Let's use this intuition to create a loss function.

### Our First Loss Function: Mean Squared Error

We would like our choice of $ \theta $ to fall close to the points in our dataset. Thus, we can define a loss function that outputs a larger value as $ \theta $ gets further away from the points in the dataset. We start with a simple loss function called the *mean squared error*. Here's the idea:

1. We select a value of $ \theta $.
2. For each value in our dataset, take the squared difference between the value and theta: $ (y_i - \theta)^2 $ . Squaring the difference in a simple way to convert negative differences into positive ones. We want to do this because if our point $ y_i = 14 $, $ \theta = 10 $ and $ \theta = 18 $ are equally far away from the point and are thus equally "bad".
3. To compute the final loss, take the average of each of the individual squared differences.

This gives us a final loss function of:


$$
\begin{aligned}
L(\theta, y_1, y_2, \ldots, y_n)
&= \text{average}\left\{ (y_1 - \theta)^2, (y_2 - \theta)^2, \ldots, (y_n - \theta)^2 \right\} \\
&= \frac{1}{n} \left((y_1 - \theta)^2 + (y_2 - \theta)^2 + \ldots + (y_n - \theta)^2 \right) \\
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\end{aligned}
$$

Creating a Python function to compute the loss is simple to do:


```python
def mse_loss(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)
```

Let's see how this loss function behaves. Suppose we have a dataset only containing one point, $ y_1 = 14 $. We can try different values of $ \theta $ and see what the loss function outputs for each value.


```python
# HIDDEN
def try_thetas(thetas, y_vals, xlims, loss_fn=mse_loss, figsize=(10, 7), cols=3):
    if not isinstance(y_vals, np.ndarray):
        y_vals = np.array(y_vals)
    rows = int(np.ceil(len(thetas) / cols))
    plt.figure(figsize=figsize)
    for i, theta in enumerate(thetas):
        ax = plt.subplot(rows, cols, i + 1)
        sns.rugplot(y_vals, height=0.1, ax=ax)
        plt.axvline(theta, linestyle='--',
                    label=rf'$ \theta = {theta} $')
        plt.title(f'Loss = {loss_fn(theta, y_vals):.2f}')
        plt.xlim(*xlims)
        plt.yticks([])
        plt.legend()
    plt.tight_layout()

try_thetas(thetas=[11, 12, 13, 14, 15, 16],
           y_vals=[14], xlims=(10, 17))
```


![png](modeling_loss_functions_files/modeling_loss_functions_8_0.png)


You can also interactively try different values of $ \theta $ below. You should understand why the loss for $ \theta = 11 $ is many times higher than the loss for $ \theta = 13 $.


```python
# HIDDEN
def try_thetas_interact(theta, y_vals, xlims, loss_fn=mse_loss):
    if not isinstance(y_vals, np.ndarray):
        y_vals = np.array(y_vals)
    plt.figure(figsize=(4, 3))
    sns.rugplot(y_vals, height=0.1)
    plt.axvline(theta, linestyle='--')
    plt.xlim(*xlims)
    plt.yticks([])
    print(f'Loss for theta = {theta}: {loss_fn(theta, y_vals):.2f}')

def mse_interact(theta, y_vals, xlims):
    plot = interactive(try_thetas_interact, theta=theta,
                       y_vals=fixed(y_vals), xlims=fixed(xlims),
                       loss_fn=fixed(mse_loss))
    plot.children[-1].layout.height = '240px'
    return plot
    
mse_interact(theta=(11, 16, 0.5), y_vals=[14], xlims=(10, 17))
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



As we hoped, our loss is larger as $ \theta $ is further away from our data and is smallest when $ \theta $ falls exactly onto our data point. Let's now see how our mean squared error behaves when we have five points instead of one. Our data this time are: $ \{ 12.1, 12.8, 14.9, 16.3, 17.2 \} $.


```python
# HIDDEN
try_thetas(thetas=[12, 13, 14, 15, 16, 17],
           y_vals=[12.1, 12.8, 14.9, 16.3, 17.2],
           xlims=(11, 18))
```


![png](modeling_loss_functions_files/modeling_loss_functions_12_0.png)


Of the values of $ \theta $ we tried $ \theta = 15 $ has the lowest loss. However, a value of $ \theta $ in between 14 and 15 might have an even lower loss than $ \theta = 15 $. See if you can find a better value of $ \theta $ using the interactive plot below.

(How might we know for certain whether we've found the best value of $ \theta $? We will tackle this issue soon.)


```python
# HIDDEN
mse_interact(theta=(12, 17, 0.2),
             y_vals=[12.1, 12.8, 14.9, 16.3, 17.2],
             xlims=(11, 18))
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



The mean squared error seems to be doing its job by penalizing values of $ \theta $ that are far away from the center of the data. Let's now see what the loss function outputs on the original dataset of tip percents. For reference, the original distribution of tip percents is plotted below:


```python
# HIDDEN
sns.distplot(tips['pcttip'], bins=np.arange(30), rug=True)
plt.xlim(0, 30)
plt.xlabel('Percent Tip Amount')
plt.ylabel('Proportion per Percent');
```


![png](modeling_loss_functions_files/modeling_loss_functions_16_0.png)


Let's try some values of $ \theta $.


```python
# HIDDEN
try_thetas(thetas=np.arange(13, 17.1, 0.5),
           y_vals=tips['pcttip'],
           xlims=(0, 30))
```


![png](modeling_loss_functions_files/modeling_loss_functions_18_0.png)


As before, we've created an interactive widget to test different values of $ \theta $.


```python
# HIDDEN
mse_interact(theta=(13, 17, 0.25),
             y_vals=tips['pcttip'],
             xlims=(0, 30))
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



It looks like the best value of $ \theta $ that we've tried so far is 16.00. This is slightly above our original blind guess of 15% tip. It appears that our waiter gets a bit more tip than we originally thought.

### A Shorthand

We have defined our first loss function, the mean squared error (MSE). It computes high loss for values of $ \theta $ that are further away from the center of the data. Mathematically, this loss function is defined as:

$$
\begin{aligned}
L(\theta, y_1, y_2, \ldots, y_n)
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\end{aligned}
$$

The loss function will compute different losses whenever we change either $ \theta $ or $ y_1, y_2, \ldots, y_n $. We've seen this happen when we tried different values of $ \theta $ and when we added new data points (changing $ y_1, y_2, \ldots, y_n $). 

As a shorthand, we can define the vector $ \textbf{y} = [ y_1, y_2, \ldots, y_n ] $. Then, we can write MSE as:

$$
\begin{aligned}
L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\end{aligned}
$$

### Finding the Best Value of $ \theta $

So far, we have found the best value of $ \theta $ by simply trying out a bunch of values and then picking the one with the least loss. Although this method works decently well, we can find a better method by using the properties of our loss function.

For the time being, let's return to our example with five points: $ \textbf{y} = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $.


```python
# HIDDEN
try_thetas(thetas=[12, 13, 14, 15, 16, 17],
           y_vals=[12.1, 12.8, 14.9, 16.3, 17.2],
           xlims=(11, 18))
```


![png](modeling_loss_functions_files/modeling_loss_functions_24_0.png)


In the plots above, we've used integer $ \theta $ values in between 12 and 17. When we change $ \theta $, the loss seems to start high (at 10.92), decrease until $ \theta = 15 $, then increase again. We can see that the loss changes as $ \theta $ changes, so let's make a plot comparing the loss to $ \theta $ for each of the six $ \theta $s we've tried.


```python
thetas = np.array([12, 13, 14, 15, 16, 17])
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
losses = [mse_loss(theta, y_vals) for theta in thetas]

plt.scatter(thetas, losses)
plt.title(r'Loss vs. $ \theta $ when $\bf{y}$$ = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
```


![png](modeling_loss_functions_files/modeling_loss_functions_26_0.png)


The scatter plot shows the downward, then upward trend that we noticed before. We can try more values of $ \theta $ to see a complete curve that shows how the loss changes as $ \theta $ changes.


```python
thetas = np.arange(12, 17.1, 0.05)
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
losses = [mse_loss(theta, y_vals) for theta in thetas]

plt.plot(thetas, losses)
plt.title(r'Loss vs. $ \theta $ when $\bf{y}$$ = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
```


![png](modeling_loss_functions_files/modeling_loss_functions_28_0.png)


The plot above shows that in fact, $ \theta = 15$ was not the best choice; a $ \theta $ of around 14.7 would have gotten a lower loss! We can use calculus to find that minimizing value of $ \theta $ exactly. First, we start with our loss function:

$$
\begin{aligned}
L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\end{aligned}
$$

And plug in our points $ \textbf{y} = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $:

$$
\begin{aligned}
L(\theta, \textbf{y})
&= \frac{1}{n} \big((12.1 - \theta)^2 + (12.8 - \theta)^2 + (14.9 - \theta)^2 + (16.3 - \theta)^2 + (17.2 - \theta)^2 \big)\\
\end{aligned}
$$

To find the value of $ \theta $ that minimizes this function, we compute the derivative with respect to $ \theta $:

$$
\begin{aligned}
\frac{\partial}{\partial \theta} L(\theta, \textbf{y})
&= \frac{1}{n} \big(-2(12.1 - \theta) - 2(12.8 - \theta) - 2(14.9 - \theta) - 2(16.3 - \theta) -2(17.2 - \theta) \big)\\
&= - \frac{2}{n} \big((12.1 - \theta) + (12.8 - \theta) + (14.9 - \theta) + (16.3 - \theta) + (17.2 - \theta) \big)
\end{aligned}
$$

Then, we find the value of $ \theta $ where the derivative is zero:

$$
\begin{aligned}
0 &= - \frac{2}{n} \big((12.1 - \theta) + (12.8 - \theta) + (14.9 - \theta) + (16.3 - \theta) + (17.2 - \theta) \big) \\
0 &= (12.1 - \theta) + (12.8 - \theta) + (14.9 - \theta) + (16.3 - \theta) + (17.2 - \theta) \\
5 \theta &= 12.1 + 12.8 + 14.9 + 16.3 + 17.2 \\
\theta &= \frac{12.1 + 12.8 + 14.9 + 16.3 + 17.2}{5} \\
\theta &= 14.66
\end{aligned}
$$

As expected, the value of $ \theta $ that minimizes the loss is between 14 and 15.

Turn your attention to the second-to-last line of the simplification above:

$$ \hat{\theta} = \frac{12.1 + 12.8 + 14.9 + 16.3 + 17.2}{5} $$

Notice that this is a familiar expression: it is the average of the five data points. Could this be a pattern for all values of $ \textbf{y} $?

### The Minimizing Value of the Mean Squared Error

We have seen that different values of $ \theta $ produce different losses when using mean squared error as our loss function. The arithmetic above hints that the value of $ \theta $ that minimizes the loss is the mean of all of the data points. To confirm this, we turn back to the definition of our loss function. Instead of plugging in points, we take the derivative with respect to $ \theta $ of the loss function as-is:

$$
\begin{aligned}
L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\frac{\partial}{\partial \theta} L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n} -2(y_i - \theta) \\
&= -\frac{2}{n} \sum_{i = 1}^{n} (y_i - \theta) \\
\end{aligned}
$$

Notice that we have left the variables $ y_i $ untouched. We are no longer working with the previous example dataset of five points; this equation can be used with any dataset with any number of points.

Now, we set the derivative equal to zero and solve for $ \theta $ to find the minimizing value of $ \theta $ as before:

$$
\begin{aligned}
-\frac{2}{n} \sum_{i = 1}^{n} (y_i - \theta) &= 0 \\
\sum_{i = 1}^{n} (y_i - \theta) &= 0 \\
\sum_{i = 1}^{n} y_i - \sum_{i = 1}^{n} \theta &= 0 \\
\sum_{i = 1}^{n} \theta &= \sum_{i = 1}^{n} y_i \\
n \cdot \theta &= y_1 + \ldots + y_n \\
\theta &= \frac{y_1 + \ldots + y_n}{n} \\
\theta &= \text{mean} (\textbf{y})
\end{aligned}
$$

Lo and behold, we see that there is a single value of $ \theta $ that gives the least MSE no matter what the dataset is. For the mean squared error, we can set $ \theta $ equal to the mean of the dataset and be confident knowing that we have minimized the loss.

### Back to the Original Dataset

We no longer have to test out different values of $ \theta $ as we did before. We can compute the mean tip percentage in one go:


```python
np.mean(tips['pcttip'])
```




    16.08025817225047




```python
# HIDDEN
sns.distplot(tips['pcttip'], bins=np.arange(30), rug=True)

plt.axvline(x=16.08, c='darkblue', linestyle='--', label=r'$ \theta = 16.08$')
plt.legend()

plt.xlim(0, 30)
plt.title('Distribution of tip percent')
plt.xlabel('Percent Tip Amount')
plt.ylabel('Proportion per Percent');
```


![png](modeling_loss_functions_files/modeling_loss_functions_34_0.png)


### Summary

First, we restricted our model to only make a single number as its prediction for all tables. Next, we assume that the waiter's dataset of tips is similar to the population distribution of tip percentages. If this assumption holds, predicting $ 16.08\% $ will give us the most accurate predictions that we can given our data.

To be more precise, we say that the model is accurate if it minimizes the squared difference between the predictions and the actual values.

Although our model is simple, it illustrates concepts that we'll see over and over again. Future chapters will introduce complicated models. Still, we will discuss each model's assumptions, define loss functions, and find the model that minimizes the loss. It is very helpful to understand this process for simple models before attempting to understand complex ones.
