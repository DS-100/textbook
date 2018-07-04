

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 7
np.set_printoptions(precision=2, suppress=True)
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

It is also a good idea to determine which of our variables are quantitative and which are categorical.

Quantitative: `Length`, `Girth`, `Height`, `Weight`, `WeightAlt`

Categorical: `BCS`, `Age`, `Sex`


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



Also looking at the barplot of `BCS`:


```python
plt.hist(donkeys['BCS'], normed=True)
plt.xlabel('BCS');
```


![png](linear_case_study_files/linear_case_study_17_0.png)


Considering that `BCS` is an indication of the health of a donkey, a `BCS` of 1 represents an extremely emaciated donkey and a `BCS` of 4.5 an overweight donkey. Also looking at the barplot, there only appear to be two donkeys with such outlier `BCS` values. Thus, it would probably be a good idea to remove these two donkeys.

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

## Train-Test Split

Before we proceed with our data analysis, we divide our data into an 80/20 split, using 80% of our data to train our model and setting aside the other 20% for evaluation of the model.


```python
X_train, X_test, y_train, y_test = train_test_split(donkeys_c.drop(['Weight'], axis=1),
                                                    donkeys_c['Weight'],
                                                    test_size=0.2,
                                                   random_state=42)
X_train.shape, X_test.shape
```




    ((431, 7), (108, 7))



Let's also create a function that we can use to evaluate our models on our test set. Recall that the metric for linear regression is mean-squared error. 


```python
def mse_test_set(predictions):
    return float(np.sum((predictions - y_test) ** 2))
```

## Exploring the Data + Data Visualization

As in any data science project, we will explore our data before attempting to fit a model to it.

First, we will examine the categorical variables with boxplots.


```python
# HIDDEN
sns.boxplot(x=X_train['BCS'], y=y_train);
```


![png](linear_case_study_files/linear_case_study_31_0.png)


It seems like median weight increases with BCS, but not linearly.


```python
# HIDDEN
sns.boxplot(x=X_train['Sex'], y=y_train,
            order = ['female', 'stallion', 'gelding']);
```


![png](linear_case_study_files/linear_case_study_33_0.png)


It seems like the sex of the donkey doesn't appear to cause much of a difference in weight.


```python
# HIDDEN
sns.boxplot(x=X_train['Age'], y=y_train, 
            order = ['<2', '2-5', '5-10', '10-15', '15-20', '>20']);
```


![png](linear_case_study_files/linear_case_study_35_0.png)


For donkeys over 5, the weight distribution is not too different.

Now, let's look at the quantitative variables. We can plot each of them against the target variable.


```python
# HIDDEN
X_train['Weight'] = y_train
sns.regplot('Length', 'Weight', X_train, fit_reg=False);
```


![png](linear_case_study_files/linear_case_study_37_0.png)



```python
# HIDDEN
sns.regplot('Girth', 'Weight', X_train, fit_reg=False);
```


![png](linear_case_study_files/linear_case_study_38_0.png)



```python
# HIDDEN
sns.regplot('Height', 'Weight', X_train, fit_reg=False);
```


![png](linear_case_study_files/linear_case_study_39_0.png)


All three of our quantitative features have a linear relationship with our target variable of `Weight`, so we will not have to perform any transformations on our input data.

It is also a good idea to see if our features are linear with each other. We plot two below: 


```python
# HIDDEN
sns.regplot('Height', 'Length', X_train, fit_reg=False);
```


![png](linear_case_study_files/linear_case_study_41_0.png)



```python
# HIDDEN
sns.regplot('Height', 'Girth', X_train, fit_reg=False);
```


![png](linear_case_study_files/linear_case_study_42_0.png)


From these plots, we can see that our predictor variables also have strong linear relationships with each other. This makes our model harder to interpret, so we should keep this in mind after we create our model.

## Simpler Linear Models

Rather than using all of our data at once, let's try to fit linear models to one or two variables first. 

Below are three simple linear regression models using just one quantitative variable. Which model appears to be the best?


```python
# HIDDEN
sns.regplot('Length', 'Weight', X_train, fit_reg=True);
```


![png](linear_case_study_files/linear_case_study_44_0.png)



```python
# HIDDEN
model = LinearRegression()
model.fit(X_train[['Length']], X_train['Weight'])
predictions = model.predict(X_test[['Length']])
print("MSE:", mse_test_set(predictions))
```

    MSE: 26052.580077025486
    


```python
sns.regplot('Girth', 'Weight', X_train, fit_reg=True);
```


![png](linear_case_study_files/linear_case_study_46_0.png)



```python
# HIDDEN
model = LinearRegression()
model.fit(X_train[['Girth']], X_train['Weight'])
predictions = model.predict(X_test[['Girth']])
print("MSE:", mse_test_set(predictions))
```

    MSE: 13248.81410593239
    


```python
sns.regplot('Height', 'Weight', X_train, fit_reg=True);
```


![png](linear_case_study_files/linear_case_study_48_0.png)



```python
# HIDDEN
model = LinearRegression()
model.fit(X_train[['Height']], X_train['Weight'])
predictions = model.predict(X_test[['Height']])
print("MSE:", mse_test_set(predictions))
```

    MSE: 36343.308584306156
    

Looking at the scatterplots and the mean-squared errors, it seems like `Girth` is the best sole predictor of `Weight` as it has the strongest linear relationship with `Weight` and the smallest mean-squared error.

Can we do better with two variables? Let's try fitting a linear model using both `Girth` and `Length`. Although it is not as easy to visualize this model, we can still look at the MSE of this model.


```python
# HIDDEN
model = LinearRegression()
model.fit(X_train[['Girth', 'Length']], X_train['Weight'])
predictions = model.predict(X_test[['Girth', 'Length']])
print("MSE:", mse_test_set(predictions))
```

    MSE: 9680.90242337725
    

Wow! Looks like our MSE went down from around 13000 with just `Girth` alone to 10000 with `Girth` and `Length`. Adding more variables seems to improve our model.

We can also use categorical variables in our model. Let's now look at a linear model using the categorical variable of `Age`. This is the plot of `Age` versus `Weight`:


```python
# HIDDEN
sns.stripplot(x='Age', y='Weight', data=X_train, order=['<2', '2-5', '5-10', '10-15', '15-20', '>20']);
```


![png](linear_case_study_files/linear_case_study_53_0.png)


Seeing how `Age` is a categorical variable, we need to introduce dummy variables in order to produce a linear regression model.


```python
# HIDDEN
just_age_and_weight = X_train[['Age', 'Weight']]
with_age_dummies = pd.get_dummies(just_age_and_weight, columns=['Age'])
model = LinearRegression()
model.fit(with_age_dummies.drop('Weight', axis=1), with_age_dummies['Weight'])

just_age_and_weight_test = X_test[['Age']]
with_age_dummies_test = pd.get_dummies(just_age_and_weight_test, columns=['Age'])
predictions = model.predict(with_age_dummies_test)
print("MSE:", mse_test_set(predictions))
```

    MSE: 41511.58282277702
    

A MSE of around 40000 is worse than what we could get using any single one of the quantitative variables, but this variable could still prove to be useful in our linear model.

Let's try to interpret this linear model. Note that every donkey that falls into an age category, say 2-5 years of age, will receive the same prediction because they all have input data which is a 1 in the 2-5 column and 0 in the rest of the age columns. Thus, we can interpret categorical variables as simply changing the constant in the model because the categorical variable separates the donkeys into groups and gives one prediction for all donkeys within that group.

Our next step is to create a final model using both our categorical variables and multiple quantitative variables.

## Transforming Variables

Recall from our boxplots that `Sex` was not a useful variable, so we will drop it. We will also remove the `WeightAlt` column because we only have its value for 31 donkeys. Also, using `get_dummies`, we transform the categorical variables `BCS` and `Age` into dummy variables so that we can include them in the model. 


```python
# HIDDEN
X_train = X_train.drop('Weight', axis=1)
```


```python
donkeys_train = X_train.drop(['Sex', 'WeightAlt'], axis=1)
donkeys_train = pd.get_dummies(donkeys_train, columns=['BCS', 'Age'])
donkeys_train.head()
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
      <th>465</th>
      <td>98</td>
      <td>113</td>
      <td>99</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>233</th>
      <td>101</td>
      <td>119</td>
      <td>101</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>106</td>
      <td>125</td>
      <td>103</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>453</th>
      <td>93</td>
      <td>120</td>
      <td>100</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>452</th>
      <td>98</td>
      <td>120</td>
      <td>108</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 15 columns</p>
</div>



Since we do not want our matrix to be over-parameterized, we should drop one category from the `BCS` and `Age` dummies.


```python
donkeys_train = donkeys_train.drop(['BCS_3.0', 'Age_5-10'], axis=1)
donkeys_train.head()
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
      <th>465</th>
      <td>98</td>
      <td>113</td>
      <td>99</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>233</th>
      <td>101</td>
      <td>119</td>
      <td>101</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>106</td>
      <td>125</td>
      <td>103</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>453</th>
      <td>93</td>
      <td>120</td>
      <td>100</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>452</th>
      <td>98</td>
      <td>120</td>
      <td>108</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 13 columns</p>
</div>



We should also add a column of biases in order to have a constant term in our model.


```python
donkeys_train = donkeys_train.assign(bias=1)
```


```python
# HIDDEN
donkeys_train = donkeys_train.reindex_axis(['bias'] + list(donkeys_train.columns[:-1]), axis=1)

```


```python
donkeys_train.head()
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
      <th>465</th>
      <td>1</td>
      <td>98</td>
      <td>113</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>233</th>
      <td>1</td>
      <td>101</td>
      <td>119</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>1</td>
      <td>106</td>
      <td>125</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>453</th>
      <td>1</td>
      <td>93</td>
      <td>120</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>452</th>
      <td>1</td>
      <td>98</td>
      <td>120</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14 columns</p>
</div>



## Multiple Linear Regression Model

We are finally ready to fit our model!

Our model looks like this:

$$
f_\hat{\theta} (x) = \hat{\theta_0} + \hat{\theta_1} (Length) + \hat{\theta_2} (Girth) + \hat{\theta_3} (Height) + ... +  \hat{\theta_{13}} (Age_>20)
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

In order to use the above functions, we need `X`, and `y`. These can both be obtained from our data frames. Remember that `X` and `y` have to be numpy matrices in order to be able to multiply them with `@` notation.

`X` consists of all columns of the data frame `donkeys_train`.


```python
X = (donkeys_train
     .as_matrix())
```

`y` is `y_train` as a matrix.


```python
y = y_train.as_matrix()
```

Now we just need to call the `minimize` function defined in a previous section.


```python
thetas = minimize(mse_cost, grad_mse_cost, X, y)
```

    theta: [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.] | cost: 23979.72
    theta: [ 0.01  0.53  0.65  0.56  0.    0.    0.    0.    0.    0.    0.    0.    0.
      0.  ] | cost: 1214.01
    theta: [-0.07  1.84  2.55 -2.87 -0.02 -0.13 -0.34  0.19  0.07  0.36  0.06 -0.22
     -0.3   0.01] | cost: 1002.77
    theta: [-0.25 -0.76  4.82 -3.07 -0.08 -0.38 -1.11  0.61  0.24  1.06  0.19 -0.66
     -0.94  0.04] | cost: 816.52
    theta: [-0.44 -0.32  4.1  -2.72 -0.14 -0.61 -1.89  1.01  0.4   1.72  0.33 -1.07
     -1.58  0.07] | cost: 494.10
    theta: [-1.54  0.87  2.04 -1.63 -0.51 -2.14 -5.64  3.24  1.39  4.47  1.28 -2.89
     -5.24  0.42] | cost: 143.54
    theta: [-2.29  0.92  1.76 -1.35 -0.8  -3.34 -7.28  4.55  2.1   5.32  1.98 -3.6
     -7.57  0.81] | cost: 132.16
    theta: [ -4.35   0.87   1.33  -0.8   -1.63  -6.75 -10.53   7.72   4.1    6.62
       3.84  -5.05 -13.52   1.95] | cost: 116.75
    theta: [ -6.22   0.77   1.16  -0.48  -2.43  -9.93 -11.93   9.94   5.98   6.74
       5.35  -5.88 -18.2    2.99] | cost: 109.21
    theta: [ -8.01   0.67   1.12  -0.32  -3.28 -12.94 -11.69  11.25   7.89   6.02
       6.5   -6.31 -21.66   3.86] | cost: 104.58
    theta: [ -9.43   0.64   1.15  -0.3   -4.01 -15.06 -10.42  11.47   9.47   5.08
       6.99  -6.5  -23.25   4.26] | cost: 102.10
    theta: [-10.74   0.65   1.17  -0.32  -4.74 -16.57  -8.98  11.08  10.96   4.32
       7.04  -6.74 -23.68   4.26] | cost: 100.54
    theta: [-12.07   0.68   1.16  -0.33  -5.48 -17.54  -8.06  10.47  12.39   3.92
       6.81  -7.08 -23.46   3.94] | cost: 99.44
    theta: [-13.98   0.71   1.15  -0.31  -6.54 -18.38  -7.63   9.72  14.29   3.7
       6.36  -7.57 -22.93   3.3 ] | cost: 98.28
    theta: [-16.8    0.73   1.12  -0.27  -8.06 -19.11  -7.88   8.96  16.86   3.52
       5.72  -8.16 -22.22   2.36] | cost: 96.94
    theta: [-20.26   0.73   1.1   -0.21  -9.88 -19.49  -8.8    8.48  19.63   3.27
       5.03  -8.57 -21.56   1.34] | cost: 95.64
    theta: [-23.43   0.72   1.11  -0.17 -11.48 -19.3   -9.85   8.5   21.67   2.93
       4.53  -8.56 -21.2    0.67] | cost: 94.63
    theta: [-26.29   0.7    1.13  -0.15 -12.85 -18.66 -10.55   8.87  22.91   2.54
       4.2   -8.23 -21.1    0.41] | cost: 93.78
    theta: [-29.94   0.69   1.16  -0.13 -14.5  -17.51 -10.82   9.52  23.88   2.12
       3.9   -7.69 -21.08   0.41] | cost: 92.78
    theta: [-35.73   0.68   1.21  -0.13 -17.01 -15.62 -10.46  10.46  24.89   1.71
       3.51  -7.03 -21.02   0.66] | cost: 91.35
    theta: [-43.77   0.7    1.24  -0.11 -20.36 -13.23  -9.33  11.45  25.84   1.55
       3.07  -6.57 -20.81   1.13] | cost: 89.63
    theta: [-51.89   0.73   1.25  -0.08 -23.54 -11.25  -7.88  11.97  26.32   1.81
       2.78  -6.69 -20.42   1.64] | cost: 88.08
    theta: [-58.29   0.76   1.24  -0.03 -25.77 -10.26  -6.82  11.89  26.13   2.3
       2.78  -7.27 -19.93   2.01] | cost: 86.85
    theta: [-64.58   0.78   1.22   0.02 -27.6   -9.93  -6.26  11.36  25.3    2.78
       3.03  -8.05 -19.29   2.26] | cost: 85.56
    theta: [-74.17   0.79   1.22   0.1  -30.04  -9.95  -6.14  10.34  23.46   3.2
       3.59  -8.93 -18.27   2.51] | cost: 83.64
    theta: [-89.43   0.8    1.26   0.21 -33.59 -10.29  -6.61   8.84  20.18   3.3
       4.47  -9.5  -16.69   2.7 ] | cost: 80.89
    theta: [-108.09    0.81    1.34    0.3   -37.61  -10.86   -7.61    7.37   16.03
        2.89    5.28   -9.04  -14.85    2.71] | cost: 77.92
    theta: [-122.53    0.81    1.42    0.35  -40.3   -11.45   -8.48    6.7    12.84
        2.25    5.45   -7.54  -13.5     2.44] | cost: 75.76
    theta: [-129.52    0.81    1.48    0.35  -41.06  -11.92   -8.74    6.85   11.4
        1.92    4.99   -5.93  -12.86    2.06] | cost: 74.56
    theta: [-132.74    0.82    1.49    0.35  -40.75  -12.33   -8.55    7.29   10.92
        2.01    4.28   -4.78  -12.52    1.7 ] | cost: 73.81
    theta: [-135.43    0.84    1.49    0.37  -39.96  -12.74   -8.17    7.76   10.7
        2.4     3.5    -4.1   -12.15    1.4 ] | cost: 73.19
    theta: [-138.99    0.85    1.48    0.4   -38.82  -13.11   -7.77    8.17   10.46
        2.9     2.81   -3.9   -11.63    1.21] | cost: 72.59
    theta: [-144.58    0.86    1.47    0.45  -37.06  -13.41   -7.41    8.56   10.13
        3.38    2.17   -4.05  -10.84    1.15] | cost: 71.82
    theta: [-153.94    0.88    1.47    0.53  -33.91  -13.52   -7.12    9.02    9.72
        3.68    1.53   -4.52   -9.63    1.23] | cost: 70.64
    theta: [-167.87    0.89    1.49    0.62  -28.63  -13.18   -6.96    9.53    9.42
        3.51    1.01   -5.17   -8.04    1.49] | cost: 68.94
    theta: [-183.07    0.91    1.55    0.69  -22.    -12.18   -6.96    9.82    9.52
        2.7     0.85   -5.58   -6.66    1.8 ] | cost: 67.09
    theta: [-193.69    0.92    1.61    0.71  -16.37  -10.84   -7.01    9.62   10.08
        1.7     1.17   -5.38   -6.18    1.87] | cost: 65.63
    theta: [-198.64    0.93    1.66    0.71  -12.94   -9.67   -6.95    8.99   10.77
        1.13    1.7    -4.73   -6.51    1.6 ] | cost: 64.68
    theta: [-200.91    0.93    1.68    0.7   -11.06   -8.89   -6.73    8.17   11.27
        1.17    2.19   -4.     -7.11    1.09] | cost: 64.07
    theta: [-202.51    0.93    1.68    0.72  -10.22   -8.59   -6.45    7.54   11.42
        1.64    2.47   -3.58   -7.49    0.59] | cost: 63.79
    theta: [-203.71    0.93    1.67    0.74   -9.98   -8.61   -6.31    7.35   11.32
        2.01    2.52   -3.55   -7.49    0.36] | cost: 63.72
    theta: [-204.25    0.93    1.67    0.74   -9.92   -8.66   -6.29    7.38   11.25
        2.11    2.52   -3.62   -7.4     0.32] | cost: 63.71
    theta: [-204.44    0.93    1.67    0.75   -9.89   -8.67   -6.29    7.41   11.23
        2.13    2.54   -3.65   -7.36    0.29] | cost: 63.71
    theta: [-204.55    0.93    1.67    0.75   -9.86   -8.67   -6.28    7.46   11.22
        2.12    2.59   -3.67   -7.32    0.25] | cost: 63.70
    theta: [-204.51    0.93    1.67    0.75   -9.86   -8.67   -6.27    7.49   11.23
        2.12    2.64   -3.66   -7.31    0.21] | cost: 63.70
    theta: [-204.43    0.93    1.67    0.75   -9.88   -8.66   -6.27    7.49   11.24
        2.11    2.66   -3.65   -7.31    0.2 ] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.66   -6.27    7.49   11.24
        2.11    2.66   -3.64   -7.32    0.2 ] | cost: 63.70
    theta: [-204.39    0.93    1.67    0.75   -9.88   -8.66   -6.27    7.49   11.25
        2.11    2.66   -3.64   -7.32    0.2 ] | cost: 63.70
    theta: [-204.38    0.93    1.67    0.75   -9.89   -8.66   -6.27    7.49   11.25
        2.12    2.67   -3.64   -7.31    0.21] | cost: 63.70
    theta: [-204.38    0.93    1.67    0.75   -9.89   -8.67   -6.27    7.49   11.25
        2.12    2.67   -3.63   -7.31    0.22] | cost: 63.70
    theta: [-204.38    0.93    1.67    0.75   -9.89   -8.67   -6.27    7.49   11.26
        2.13    2.68   -3.62   -7.3     0.23] | cost: 63.70
    theta: [-204.39    0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    theta: [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23] | cost: 63.70
    

Looks like gradient descent converged to those theta values! Thus, our linear model is:

$y = -204.4 + 0.93x_1 + ... -7.29x_{13} - 0.64x_{13}$

Let's compare this equation that we obtained to the one we would get if we had used `sklearn`'s LinearRegression model instead.


```python
model = LinearRegression(fit_intercept=False) # We already accounted for it with the bias column
model.fit(X[:, :14], y)
print("Coefficients", model.coef_)
```

    Coefficients [-204.4     0.93    1.67    0.75   -9.88   -8.67   -6.27    7.49   11.26
        2.13    2.69   -3.62   -7.29    0.23]
    

The coefficients look pretty similar! Our homemade functions create the same model as an established Python package!

We successfully fit a linear model to our donkey data! Nice!

## Evaluating our Model

Our next step is to evaluate our model's performance on the test set. We need to perform the same data pre-processing steps on the test set as we did on the training set before we can pass it into our model.


```python
donkeys_test = X_test.drop(['Sex', 'WeightAlt'], axis=1)
donkeys_test = pd.get_dummies(donkeys_test, columns=['BCS', 'Age'])
donkeys_test = donkeys_test.drop(['BCS_3.0', 'Age_5-10'], axis=1)
donkeys_test = donkeys_test.assign(bias=1)
```


```python
# HIDDEN
donkeys_test = donkeys_test.reindex_axis(['bias'] + list(donkeys_test.columns[:-1]), axis=1)
```

We obtain `X` to pass into `predict` of our `LinearRegression` model:


```python
X = (donkeys_test
     .as_matrix())
predictions = model.predict(X)
```

With these predictions, we can make a residual plot:


```python
# HIDDEN
y = y_test.as_matrix()
resid = y - predictions
resid_prop = resid / y
plt.scatter(np.arange(len(resid_prop)), resid_prop, s=15)
plt.axhline(0)
plt.title('Residual proportions (resid / actual Weight)')
plt.xlabel('Index of row in data')
plt.ylabel('Error proportion');
```


![png](linear_case_study_files/linear_case_study_84_0.png)


Looks like our model does pretty well! The residual proportions indicate that our predictions are mostly within 15% of the correct value. Let's also look at the mean squared error:


```python
mse_test_set(predictions)
```




    7287.878537254738



Wow, look at that low mean squared error! We did it!
