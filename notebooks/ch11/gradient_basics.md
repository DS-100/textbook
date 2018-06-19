
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Cost-Minimization-Using-a-Program" data-toc-modified-id="Cost-Minimization-Using-a-Program-1">Cost Minimization Using a Program</a></span></li><li><span><a href="#Issues-with-simple_minimize" data-toc-modified-id="Issues-with-simple_minimize-2">Issues with <code>simple_minimize</code></a></span></li></ul></div>


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
def mse_cost(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def points_and_cost(y_vals, xlim, cost_fn):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    costs = [cost_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(*xlim)
    plt.title('Points')
    plt.xlabel('Tip Percent')
    
    ax = plt.subplot(122)
    plt.plot(thetas, costs)
    plt.xlim(*xlim)
    plt.title(cost_fn.__name__)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    plt.legend()
```

## Cost Minimization Using a Program

Let us return to our constant model:

$$
\theta = C
$$

We will use the mean squared error cost function:

$$
\begin{aligned}
L(\theta, y)
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2\\
\end{aligned}
$$

For simplicity, we will use the dataset $ y = [ 12, 13, 15, 16, 17 ] $. We know from our analytical approach in a previous chapter that the minimizing $ \theta $ for the MSE cost is $ \text{mean}(y) = 14.6 $. Let's see whether we can find the same value by writing a program.

If we write the program well, we will be able to use the same program on any cost function in order to find the minimizing value of $ \theta $, including the mathematically complicated Huber cost:

$$
L_\alpha(\theta, y) = \frac{1}{n} \sum_{i=1}^n \begin{cases}
    \frac{1}{2}(y_i - \theta)^2 &  | y_i - \theta | \le \alpha \\
    \alpha ( |y_i - \theta| - \frac{1}{2}\alpha ) & \text{otherwise}
\end{cases}
$$

First, we create a rug plot of the data points. To the right of the rug plot we plot the MSE cost for different values of $ \theta $.


```python
# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
points_and_cost(pts, (11, 18), mse_cost)
```


![png](gradient_basics_files/gradient_basics_4_0.png)


How might we write a program to automatically find the minimizing value of $ \theta $? The simplest method is to compute the cost for many values $ \theta $. Then, we can return the $ \theta $ value that resulted in the least cost.

We define a function called `simple_minimize` that takes in a cost function, an array of data points, and an array of $ \theta $ values to try.


```python
def simple_minimize(cost_fn, dataset, thetas):
    '''
    Returns the value of theta in thetas that produces the least cost
    on a given dataset.
    '''
    costs = [cost_fn(theta, dataset) for theta in thetas]
    return thetas[np.argmin(costs)]
```

Then, we can define a function to compute the MSE cost and pass it into `simple_minimize`.


```python
def mse_cost(theta, dataset):
    return np.mean((dataset - theta) ** 2)

dataset = np.array([12, 13, 15, 16, 17])
thetas = np.arange(12, 18, 0.1)

simple_minimize(mse_cost, dataset, thetas)
```




    14.599999999999991



This is close to the expected value:


```python
# Compute the minimizing theta using the analytical formula
np.mean(dataset)
```




    14.6



Now, we can define a function to compute the Huber cost and plot the cost against $ \theta $.


```python
def huber_cost(theta, dataset, alpha = 1):
    d = np.abs(theta - dataset)
    return np.mean(
        np.where(d < alpha,
                 (theta - dataset)**2 / 2.0,
                 alpha * (d - alpha / 2.0))
    )
```


```python
# HIDDEN
points_and_cost(pts, (11, 18), huber_cost)
```


![png](gradient_basics_files/gradient_basics_13_0.png)


Although we can see that the minimizing value of $ \theta $ should be close to 15, we do not have an analytical method of finding $ \theta $ directly for the Huber cost. Instead, we can use our `simple_minimize` function.


```python
simple_minimize(huber_cost, dataset, thetas)
```




    14.999999999999989



Now, we can return to our original dataset of tip percentages and find the best value for $ \theta $ using the Huber cost.


```python
tips = sns.load_dataset('tips')
tips['pcttip'] = tips['tip'] / tips['total_bill'] * 100
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>pcttip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>5.944673</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>16.054159</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>16.658734</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>13.978041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>14.680765</td>
    </tr>
  </tbody>
</table>
</div>




```python
# HIDDEN
points_and_cost(tips['pcttip'], (11, 20), huber_cost)
```


![png](gradient_basics_files/gradient_basics_18_0.png)



```python
simple_minimize(huber_cost, tips['pcttip'], thetas)
```




    15.499999999999988



We can see that using the Huber cost gives us $ \theta = 15.5 $. We can now compare the minimizing $ \theta $ values for MSE cost, mean absolute cost, and Huber cost.


```python
print(f"          MSE cost: theta = {tips['pcttip'].mean():.2f}")
print(f"Mean Absolute cost: theta = {tips['pcttip'].median():.2f}")
print(f"        Huber cost: theta = 15.50")
```

              MSE cost: theta = 16.08
    Mean Absolute cost: theta = 15.48
            Huber cost: theta = 15.50


We can see that the Huber cost is closer to the mean absolute cost since it is less affected by the outliers on the right side of the tip percentage distribution:


```python
sns.distplot(tips['pcttip'], bins=50);
```


![png](gradient_basics_files/gradient_basics_23_0.png)


## Issues with `simple_minimize`

Although `simple_minimize` allows us to minimize cost functions, it has some flaws that make it unsuitable for general purpose use. It's primary issue is that it only works with predetermined values of $ \theta $ to test. For example, in this code snippet we used above, we had to manually define $ \theta $ values in between 12 and 18.

```python
dataset = np.array([12, 13, 15, 16, 17])
thetas = np.arange(12, 18, 0.1)

simple_minimize(mse_cost, dataset, thetas)
```

How did we know to examine the range between 12 and 18? We had to inspect the plot of the cost function manually and see that there was a minima in that range. This process becomes impractical as we add extra complexity to our models. In addition, we manually specified a step size of 0.1 in the code above. However, if the optimal value of $ \theta $ was 12.043, our `simple_minimize` function would round to 12.00, the nearest multiple of 0.1. 

We can solve both of these issues at once by using a method called *gradient descent*.
