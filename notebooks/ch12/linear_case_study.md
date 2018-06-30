

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
%matplotlib inline

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 7
np.set_printoptions(precision=2, suppress=True)
```


```python
# HIDDEN
"""
def minimize(cost_fn, grad_cost_fn, X, y, alpha=0.2, progress=True):
    '''
    Uses gradient descent to minimize cost_fn. Returns the minimizing value of
    theta once theta changes less than 0.001 between iterations.
    '''
    theta = np.zeros(X.shape[1])
    old_cost = 10000000
    while True:
        if progress:
            print(f'theta: {theta} | cost: {cost_fn(theta, X, y):.2f}')
        gradient = grad_cost_fn(theta, X, y)
        new_theta = theta - alpha * gradient
        cost = cost_fn(theta, X, y)
        if abs(old_cost - cost) < 10:
            return new_theta
        old_cost = cost
        
        theta = new_theta
        
"""
        
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

## Linear Regression Case Study

In this section, we perform an end-to-end case study of the application of a linear regression model to a dataset. The dataset we will be working with is a record of various attributes, such as length and girth, of donkeys.

Our task: predict a donkey's weight using linear regression.

## Preliminary Data Overview

We will begin by reading in the dataset and taking a quick peek at its contents.


```python
donkeys = pd.read_csv("donkeys.csv")
donkeys.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BCS</th>
      <th>Age</th>
      <th>Sex</th>
      <th>...</th>
      <th>Height</th>
      <th>Weight</th>
      <th>WeightAlt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>&lt;2</td>
      <td>stallion</td>
      <td>...</td>
      <td>90</td>
      <td>77</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.5</td>
      <td>&lt;2</td>
      <td>stallion</td>
      <td>...</td>
      <td>94</td>
      <td>100</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>&lt;2</td>
      <td>stallion</td>
      <td>...</td>
      <td>95</td>
      <td>74</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>&lt;2</td>
      <td>female</td>
      <td>...</td>
      <td>96</td>
      <td>116</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.5</td>
      <td>&lt;2</td>
      <td>female</td>
      <td>...</td>
      <td>91</td>
      <td>91</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 8 columns</p>
</div>



Now, we will look at how large our dataset is.

It's always a good idea to look at _how much_ data we have by looking at the dimensions of the dataset. An example of how this might be useful: if our data has a large number of observations, it would not be a good idea to print out the whole dataframe.


```python
donkeys.shape
```




    (544, 8)



The dataset is relatively small, with only 544 rows of observations and 8 columns. Let's look at what columns we have available to us.


```python
donkeys.columns.values
```




    array(['BCS', 'Age', 'Sex', 'Length', 'Girth', 'Height', 'Weight',
           'WeightAlt'], dtype=object)



Our analysis can be guided by a good understanding of our data, so we should aim to understand what each of these columns represent. A few of these columns are self-explanatory, but others require a little more explanation:

- `BCS`: Body Condition Score (a physical health rating)
- `Girth`: the measurement around the middle of the donkey
- `WeightAlt`: 31 donkeys within our data frame were weighed twice in order to check the accuracy of the scale. The second weighing is in `WeightAlt`.


## Data Cleaning

In this section, we will check the data for any abnormalities that we have to deal with.

By examining `WeightAlt` more closely, we can make sure that the scale is accurate by taking the difference between the two different weighings and plotting them.


```python
difference = donkeys['WeightAlt'] - donkeys['Weight']
sns.distplot(difference.dropna());
```


![png](linear_case_study_files/linear_case_study_11_0.png)


The measurements are all within 1 kg of each other, which seems reasonable.

Next, we can look for unusual values that might indicate errors or other problems. We can use the quantile function in order to detect anomalous values.


```python
donkeys.quantile([0.005, 0.995])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BCS</th>
      <th>Length</th>
      <th>Girth</th>
      <th>Height</th>
      <th>Weight</th>
      <th>WeightAlt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.005</th>
      <td>1.5</td>
      <td>71.145</td>
      <td>90.000</td>
      <td>89.0</td>
      <td>71.715</td>
      <td>98.75</td>
    </tr>
    <tr>
      <th>0.995</th>
      <td>4.0</td>
      <td>111.000</td>
      <td>131.285</td>
      <td>112.0</td>
      <td>214.000</td>
      <td>192.80</td>
    </tr>
  </tbody>
</table>
</div>



For each of these numerical columns, we can look at which rows fall outside of these quantiles and what values they take on. Consider that we want our model to apply to only healthy and mature donkeys.


```python
donkeys[(donkeys['BCS'] < 1.5) | (donkeys['BCS'] > 4)]['BCS']
```




    291    4.5
    445    1.0
    Name: BCS, dtype: float64



Considering that `BCS` is an indication of the health of a donkey, a `BCS` of 1 represents an extremely emaciated donkey and a `BCS` of 4.5 an overweight donkey. Thus, it would probably be a good idea to remove these two donkeys.

---


```python
donkeys[(donkeys['Length'] < 71.145) | (donkeys['Length'] > 111)]['Length']
```




    8       46
    22      68
    26      69
    216    112
    Name: Length, dtype: int64




```python
donkeys[(donkeys['Girth'] < 90) | (donkeys['Girth'] > 131.285)]['Girth']
```




    8       66
    239    132
    283    134
    523    134
    Name: Girth, dtype: int64




```python
donkeys[(donkeys['Height'] < 89) | (donkeys['Height'] > 112)]['Height']
```




    8       71
    22      86
    244    113
    523    116
    Name: Height, dtype: int64



For `Girth`, `Height`, and `Length`, the donkey in row 8 seems to have a much smaller value than the cut-off while the other anomalous donkeys are close to the cut-off and likely do not need to be removed.

---


```python
donkeys[(donkeys['Weight'] < 71.715) | (donkeys['Weight'] > 214)]['Weight']
```




    8       27
    26      65
    50      71
    291    227
    523    230
    Name: Weight, dtype: int64



The first 2 and last 2 donkeys in the list are far off from the cut-off and most likely should be removed. The middle donkey can be included.

---

Since `WeightAlt` closely corresponds to `Weight`, we do not have to check this column for anomalies. Summarizing what we have learned, here is how we want to filter our donkeys:

- Keep donkeys with `BCS` in the range 1.5 and 4
- Keep donkeys with `Weight` between 71 and 214  



```python
donkeys_c = donkeys[(donkeys['BCS'] >= 1.5) & (donkeys['BCS'] <= 4) &
                         (donkeys['Weight'] >= 71) & (donkeys['Weight'] <= 214)]
```

## Exploring the Data + Data Visualization

As in any data science project, we will explore our data before attempting to fit a model to it.

First, we will examine the categorical variables with boxplots.


```python
# HIDDEN
sns.boxplot(x=donkeys_c['BCS'], y=donkeys_c['Weight']);
```


![png](linear_case_study_files/linear_case_study_25_0.png)


It seems like median weight increases with BCS, but not linearly.


```python
# HIDDEN
sns.boxplot(x=donkeys_c['Sex'], y=donkeys_c['Weight'],
            order = ['female', 'stallion', 'gelding']);
```


![png](linear_case_study_files/linear_case_study_27_0.png)


It seems like the sex of the donkey doesn't appear to cause much of a difference in weight.


```python
# HIDDEN
sns.boxplot(x=donkeys_c['Age'], y=donkeys_c['Weight'], 
            order = ['<2', '2-5', '5-10', '10-15', '15-20', '>20']);
```


![png](linear_case_study_files/linear_case_study_29_0.png)


For donkeys over 5, the weight distribution is not too different.

Now, let's look at the quantitative variables. We can plot each of them against the target variable.


```python
# HIDDEN
sns.regplot('Length', 'Weight', donkeys_c);
```


![png](linear_case_study_files/linear_case_study_31_0.png)



```python
# HIDDEN
sns.regplot('Girth', 'Weight', donkeys_c);
```


![png](linear_case_study_files/linear_case_study_32_0.png)



```python
# HIDDEN
sns.regplot('Height', 'Weight', donkeys_c);
```


![png](linear_case_study_files/linear_case_study_33_0.png)


All three of our quantitative features have a linear relationship with our target variable of `Weight`, so we will not have to perform any transformations on our input data.

It is also a good idea to see if our features are linear with each other. We plot two below: 


```python
# HIDDEN
sns.regplot('Height', 'Length', donkeys_c);
```


![png](linear_case_study_files/linear_case_study_35_0.png)



```python
# HIDDEN
sns.regplot('Height', 'Girth', donkeys_c);
```


![png](linear_case_study_files/linear_case_study_36_0.png)


From these plots, we can see that our predictor variables also have strong linear relationships with each other. This makes our model harder to interpret, so we should keep this in mind after we create our model.

## Transforming Variables

Recall from our boxplots that `Sex` was not a useful variable, so we will drop it. We will also remove the `WeightAlt` column because we only have its value for 31 donkeys. Also, using `get_dummies`, we transform the categorical variables `BCS` and `Age` into dummy variables so that we can include them in the model. 


```python
donkeys_c = donkeys_c.drop(['Sex', 'WeightAlt'], axis=1)
donkeys_c = pd.get_dummies(donkeys_c, columns=['BCS', 'Age'])
donkeys_c.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>Girth</th>
      <th>Height</th>
      <th>...</th>
      <th>Age_5-10</th>
      <th>Age_&lt;2</th>
      <th>Age_&gt;20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78</td>
      <td>90</td>
      <td>90</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>91</td>
      <td>97</td>
      <td>94</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74</td>
      <td>93</td>
      <td>95</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87</td>
      <td>109</td>
      <td>96</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79</td>
      <td>98</td>
      <td>91</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 16 columns</p>
</div>



Since we do not want our matrix to be over-parameterized, we should drop one category from the `BCS` and `Age` dummies.


```python
donkeys_c = donkeys_c.drop(['BCS_3.0', 'Age_5-10'], axis=1)
donkeys_c.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>Girth</th>
      <th>Height</th>
      <th>...</th>
      <th>Age_2-5</th>
      <th>Age_&lt;2</th>
      <th>Age_&gt;20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78</td>
      <td>90</td>
      <td>90</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>91</td>
      <td>97</td>
      <td>94</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74</td>
      <td>93</td>
      <td>95</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87</td>
      <td>109</td>
      <td>96</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79</td>
      <td>98</td>
      <td>91</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14 columns</p>
</div>



We should also add a column of biases in order to have a constant term in our model.


```python
donkeys_c = donkeys_c.assign(bias=1)
```


```python
# HIDDEN
donkeys_c = donkeys_c.reindex_axis(['bias'] + list(donkeys_c.columns[:-1]), axis=1)

```


```python
donkeys_c.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bias</th>
      <th>Length</th>
      <th>Girth</th>
      <th>...</th>
      <th>Age_2-5</th>
      <th>Age_&lt;2</th>
      <th>Age_&gt;20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>78</td>
      <td>90</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>91</td>
      <td>97</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>74</td>
      <td>93</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>87</td>
      <td>109</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>79</td>
      <td>98</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 15 columns</p>
</div>



## Linear Regression Model

We are finally ready to fit our model!

Our model looks like this:

$$
f_\hat{\theta} (x) = \hat{\theta_0} + \hat{\theta_1} (Length) + \hat{\theta_2} (Girth) + \hat{\theta_3} (Height) + ... +  \hat{\theta_{14}} (Age_>20)
$$

Remember, the $\hat{\theta_0}$ is accounted for due to the bias column in our dataframe.

Here are the functions we defined earlier in the multiple linear regression section, which we will use again:


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

In order to use the above functions, we need `X`, and `y`. These can both be obtained from our dataframe. Remember that `X` and `y` have to be numpy matrices in order to be able to multiply them with `@` notation.

`X` consists of all columns of the dataframe except for the target column, `Weight`. 


```python
X = (donkeys_c
     .drop(['Weight'], axis=1)
     .as_matrix())
```

`y` consists of the `Weight` column of our dataset.


```python
y = donkeys_c['Weight'].as_matrix()
```

Now we just need to call the `minimize` function defined in a previous section.


```python
thetas = minimize(mse_cost, grad_mse_cost, X, y)
```

    theta: [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.] | cost: 23830.83
    theta: [ 0.01  0.53  0.65  0.56  0.    0.    0.    0.    0.    0.    0.    0.    0.
      0.  ] | cost: 1251.65
    theta: [-0.07  2.    2.34 -2.78 -0.03 -0.1  -0.33  0.17  0.07  0.32  0.06 -0.19
     -0.31  0.02] | cost: 1038.97
    theta: [-0.25 -0.75  5.1  -3.39 -0.09 -0.29 -1.11  0.55  0.26  0.98  0.2  -0.59
     -0.97  0.05] | cost: 882.01
    theta: [-0.42 -0.45  4.52 -3.08 -0.14 -0.45 -1.77  0.86  0.43  1.48  0.35 -0.9
     -1.53  0.08] | cost: 600.47
    theta: [-1.25  0.49  2.84 -2.14 -0.43 -1.34 -4.54  2.36  1.29  3.36  1.13 -2.13
     -4.34  0.33] | cost: 211.62
    theta: [-2.45  0.98  1.76 -1.4  -0.9  -2.75 -7.44  4.27  2.56  5.01  2.25 -3.41
     -8.12  0.84] | cost: 141.47
    theta: [ -3.23   0.96   1.61  -1.21  -1.26  -3.77  -8.41   5.27   3.41   5.33
       2.94  -3.9  -10.27   1.26] | cost: 134.37
    theta: [ -4.99   0.88   1.41  -0.88  -2.12  -6.14  -9.83   7.27   5.38   5.58
       4.36  -4.81 -14.75   2.19] | cost: 122.79
    theta: [ -8.49   0.72   1.14  -0.38  -3.92 -10.79 -11.17  10.53   9.33   5.38
       6.78  -6.35 -22.52   3.9 ] | cost: 112.96
    theta: [ -9.6    0.7    1.12  -0.33  -4.59 -12.15 -10.54  10.94  10.62   4.89
       7.13  -6.68 -23.91   4.22] | cost: 111.01
    theta: [-11.6    0.69   1.14  -0.31  -5.88 -14.29  -8.86  11.05  12.93   3.88
       7.23  -7.22 -25.14   4.42] | cost: 108.17
    theta: [-13.64   0.7    1.15  -0.32  -7.23 -15.95  -7.4   10.74  15.17   3.04
       6.9   -7.75 -25.34   4.21] | cost: 106.52
    theta: [-15.46   0.72   1.16  -0.32  -8.43 -16.89  -6.83  10.31  16.99   2.59
       6.36  -8.18 -24.9    3.7 ] | cost: 105.29
    theta: [-18.63   0.74   1.16  -0.3  -10.47 -17.93  -6.79   9.65  19.88   2.09
       5.35  -8.81 -23.93   2.63] | cost: 103.38
    theta: [-23.09   0.76   1.14  -0.24 -13.25 -18.7   -7.74   8.99  23.45   1.64
       4.05  -9.39 -22.63   1.09] | cost: 101.44
    theta: [-25.76   0.75   1.13  -0.2  -14.78 -18.52  -8.85   8.9   24.99   1.5    3.5
      -9.39 -22.11   0.31] | cost: 100.43
    theta: [-28.32   0.74   1.12  -0.15 -16.1  -17.74  -9.92   9.12  25.75   1.42
       3.29  -9.03 -21.97  -0.14] | cost: 99.51
    theta: [-31.4    0.73   1.13  -0.11 -17.54 -16.55 -10.56   9.52  26.07   1.35
       3.27  -8.46 -22.04  -0.34] | cost: 98.51
    theta: [-36.89   0.71   1.15  -0.08 -20.   -14.57 -10.74  10.1   26.35   1.26
       3.36  -7.66 -22.15  -0.41] | cost: 96.93
    theta: [-46.75   0.71   1.19  -0.03 -24.28 -11.51 -10.02  10.85  26.71   1.19
       3.5   -6.74 -22.09  -0.33] | cost: 94.60
    theta: [-57.15   0.73   1.23   0.01 -28.56  -8.95  -8.5   11.24  26.92   1.28
       3.63  -6.46 -21.65  -0.07] | cost: 92.49
    theta: [-65.83   0.75   1.25   0.04 -31.73  -7.59  -6.9   11.17  26.75   1.51
       3.74  -6.95 -20.85   0.33] | cost: 90.78
    theta: [-72.09   0.77   1.26   0.07 -33.44  -7.41  -5.96  10.78  26.09   1.76
       3.88  -7.81 -19.89   0.78] | cost: 89.40
    theta: [-79.41   0.79   1.26   0.12 -34.78  -7.87  -5.59  10.15  24.66   1.99
       4.12  -8.85 -18.58   1.39] | cost: 87.70
    theta: [-90.71   0.8    1.28   0.21 -36.31  -8.87  -5.97   9.3   22.     2.14
       4.5   -9.77 -16.73   2.23] | cost: 85.30
    theta: [-106.81    0.81    1.3     0.33  -38.15  -10.25   -7.18    8.37   18.09
        2.1     4.95   -9.88  -14.52    3.16] | cost: 82.31
    theta: [-123.97    0.81    1.35    0.44  -39.8   -11.61   -8.64    7.7    14.11
        1.94    5.18   -8.67  -12.66    3.73] | cost: 79.49
    theta: [-136.75    0.81    1.4     0.51  -40.61  -12.58   -9.45    7.45   11.53
        1.88    5.02   -6.62  -11.7     3.66] | cost: 77.46
    theta: [-143.91    0.82    1.43    0.54  -40.45  -13.16   -9.34    7.45   10.56
        2.07    4.61   -4.78  -11.39    3.16] | cost: 76.16
    theta: [-148.08    0.83    1.45    0.54  -39.59  -13.51   -8.69    7.5    10.46
        2.47    4.15   -3.59  -11.25    2.55] | cost: 75.24
    theta: [-152.11    0.84    1.46    0.55  -38.09  -13.73   -7.85    7.54   10.65
        2.99    3.77   -2.97  -10.96    1.94] | cost: 74.35
    theta: [-158.72    0.86    1.48    0.58  -35.41  -13.76   -6.87    7.59   10.94
        3.62    3.48   -2.67  -10.31    1.24] | cost: 73.09
    theta: [-170.52    0.88    1.5     0.63  -30.71  -13.34   -5.85    7.73   11.36
        4.21    3.34   -2.65   -9.11    0.43] | cost: 71.11
    theta: [-190.66    0.91    1.56    0.73  -22.83  -11.97   -4.97    8.03   12.03
        4.47    3.47   -2.85   -7.17   -0.47] | cost: 68.21
    theta: [-210.17    0.93    1.63    0.83  -15.25   -9.94   -4.94    8.35   12.77
        3.89    3.86   -3.07   -5.53   -0.85] | cost: 65.73
    theta: [-218.49    0.93    1.68    0.86  -11.99   -8.43   -5.55    8.43   13.26
        2.91    4.1    -3.07   -5.15   -0.54] | cost: 64.59
    theta: [-218.18    0.92    1.69    0.85  -12.03   -7.96   -5.97    8.26   13.44
        2.35    4.01   -2.94   -5.49   -0.09] | cost: 64.30
    theta: [-217.06    0.92    1.69    0.84  -12.38   -7.85   -6.11    8.01   13.54
        2.15    3.75   -2.84   -5.83    0.24] | cost: 64.18
    theta: [-216.73    0.92    1.7     0.84  -12.4    -7.77   -6.11    7.64   13.63
        2.1     3.33   -2.77   -6.12    0.57] | cost: 64.09
    theta: [-217.54    0.93    1.7     0.84  -12.04   -7.71   -6.04    7.4    13.67
        2.18    3.     -2.79   -6.19    0.72] | cost: 64.06
    theta: [-218.31    0.93    1.7     0.85  -11.75   -7.68   -5.99    7.35   13.65
        2.25    2.89   -2.85   -6.14    0.72] | cost: 64.05
    theta: [-218.57    0.93    1.7     0.85  -11.66   -7.67   -5.98    7.36   13.63
        2.27    2.86   -2.89   -6.12    0.69] | cost: 64.05
    theta: [-218.75    0.93    1.69    0.85  -11.61   -7.66   -5.98    7.39   13.58
        2.26    2.81   -2.94   -6.12    0.65] | cost: 64.04
    theta: [-218.83    0.93    1.69    0.85  -11.58   -7.64   -5.98    7.44   13.52
        2.21    2.72   -3.02   -6.16    0.57] | cost: 64.04
    theta: [-218.74    0.93    1.69    0.85  -11.62   -7.63   -5.98    7.48   13.46
        2.13    2.61   -3.08   -6.24    0.5 ] | cost: 64.04
    theta: [-218.58    0.93    1.69    0.85  -11.67   -7.62   -5.98    7.48   13.43
        2.07    2.55   -3.11   -6.3     0.46] | cost: 64.04
    theta: [-218.51    0.93    1.7     0.85  -11.69   -7.62   -5.97    7.46   13.43
        2.05    2.55   -3.11   -6.33    0.44] | cost: 64.04
    theta: [-218.49    0.93    1.7     0.85  -11.7    -7.62   -5.96    7.44   13.43
        2.04    2.55   -3.11   -6.34    0.42] | cost: 64.04
    theta: [-218.51    0.93    1.7     0.85  -11.69   -7.62   -5.96    7.42   13.43
        2.04    2.56   -3.11   -6.36    0.4 ] | cost: 64.04
    theta: [-218.54    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.42
        2.04    2.57   -3.12   -6.36    0.39] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    theta: [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38] | cost: 64.04
    

Looks like gradient descent converged to those theta values! Thus, our linear model is:

$y = -218.56 + 0.93x_1 + ... -6.36x_{13} + 0.38x_{14}$

Let's compare this equation that we obtained to the one we would get if we had used `sklearn`'s LinearRegression model instead. The LinearRegression model automatically includes a column of all 1's in our data, so we can get rid of our column of 1's when we pass `X` into the `fit` function.


```python
len(donkeys_c.columns.values)
```




    15




```python
model = LinearRegression(fit_intercept=False) # We already accounted for it with the bias column
model.fit(X[:, :14], y)
print("Coefficients", model.coef_)
```

    Coefficients [-218.56    0.93    1.7     0.85  -11.68   -7.61   -5.96    7.41   13.41
        2.04    2.57   -3.12   -6.36    0.38]
    

The coefficients look pretty similar! Our homemade functions create the same model as an established Python package!

We successfully fit a linear model to our donkey data! Nice!
