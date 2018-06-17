
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#The-Walmart-dataset" data-toc-modified-id="The-Walmart-dataset-1">The Walmart dataset</a></span></li><li><span><a href="#Fitting-a-Model-Using-Scikit-Learn" data-toc-modified-id="Fitting-a-Model-Using-Scikit-Learn-2">Fitting a Model Using Scikit-Learn</a></span></li><li><span><a href="#The-One-Hot-Encoding" data-toc-modified-id="The-One-Hot-Encoding-3">The One-Hot Encoding</a></span></li><li><span><a href="#One-Hot-Encoding-in-Scikit-Learn" data-toc-modified-id="One-Hot-Encoding-in-Scikit-Learn-4">One-Hot Encoding in Scikit-Learn</a></span></li><li><span><a href="#Fitting-a-Model-Using-the-Transformed-Data" data-toc-modified-id="Fitting-a-Model-Using-the-Transformed-Data-5">Fitting a Model Using the Transformed Data</a></span></li><li><span><a href="#Model-Diagnosis" data-toc-modified-id="Model-Diagnosis-6">Model Diagnosis</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-7">Summary</a></span></li></ul></div>


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
from IPython.display import display

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
```

## The Walmart dataset

In 2014, Walmart released some of its sales data as part of a competition to predict the weekly sales of its stores. We've taken a subset of their data and loaded it below.


```python
walmart = pd.read_csv('walmart.csv')
walmart
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
      <th>Date</th>
      <th>Weekly_Sales</th>
      <th>IsHoliday</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>Unemployment</th>
      <th>MarkDown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-05</td>
      <td>24924.50</td>
      <td>No</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>8.106</td>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-02-12</td>
      <td>46039.49</td>
      <td>Yes</td>
      <td>38.51</td>
      <td>2.548</td>
      <td>8.106</td>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-02-19</td>
      <td>41595.55</td>
      <td>No</td>
      <td>39.93</td>
      <td>2.514</td>
      <td>8.106</td>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2012-10-12</td>
      <td>22764.01</td>
      <td>No</td>
      <td>62.99</td>
      <td>3.601</td>
      <td>6.573</td>
      <td>MarkDown2</td>
    </tr>
    <tr>
      <th>141</th>
      <td>2012-10-19</td>
      <td>24185.27</td>
      <td>No</td>
      <td>67.97</td>
      <td>3.594</td>
      <td>6.573</td>
      <td>MarkDown2</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2012-10-26</td>
      <td>27390.81</td>
      <td>No</td>
      <td>69.16</td>
      <td>3.506</td>
      <td>6.573</td>
      <td>MarkDown1</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 7 columns</p>
</div>



The data contains several interesting features, including whether a week contained a holiday (`IsHoliday`), the unemployment rate that week (`Unemployment`), and which special deals the store offered that week (`MarkDown`).

Our goal is to create a model that predicts the `Weekly_Sales` variable using the other variables in our data. Using a linear regression model we directly can use the `Temperature`, `Fuel_Price`, and `Unemployment` columns because they contain numerical data.

## Fitting a Model Using Scikit-Learn

In previous sections we have seen how to take the gradient of the cost function and use gradient descent to fit a model. To do this, we had to define Python functions for our model, the cost function, the gradient of the cost function, and the gradient descent algorithm. While this was important to demonstrate how the concepts work, in this section we will instead use a machine learning library called [`scikit-learn`](http://scikit-learn.org/) which allows us to fit a model with less code.

For example, to fit a multiple linear regression model using the numerical columns in the Walmart dataset, we first create a two-dimensional NumPy array containing the variables used for prediction and a one-dimensional array containing the values we want to predict:


```python
numerical_columns = ['Temperature', 'Fuel_Price', 'Unemployment']
X = walmart[numerical_columns].as_matrix()
X
```




    array([[ 42.31,   2.57,   8.11],
           [ 38.51,   2.55,   8.11],
           [ 39.93,   2.51,   8.11],
           ..., 
           [ 62.99,   3.6 ,   6.57],
           [ 67.97,   3.59,   6.57],
           [ 69.16,   3.51,   6.57]])




```python
y = walmart['Weekly_Sales'].as_matrix()
y
```




    array([ 24924.5 ,  46039.49,  41595.55, ...,  22764.01,  24185.27,
            27390.81])



Then, we import the `LinearRegression` class from `scikit-learn` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)), instantiate it, and call the `fit` method using `X` to predict `y`.

Note that previously we had to manually add a column of all $1$'s to the `X` matrix in order to conduct linear regression with an intercept. This time, `scikit-learn` will take care of the intercept column behind the scenes, saving us some work.


```python
from sklearn.linear_model import LinearRegression

simple_classifier = LinearRegression()
simple_classifier.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



We are done! When we called `.fit`, `scikit-learn` found the linear regression parameters that minimized the least squares cost function. We can see the parameters below:


```python
simple_classifier.coef_, simple_classifier.intercept_
```




    (array([ -332.22,  1626.63,  1356.87]), 29642.700510138635)



To calculate the mean squared cost, we can ask the classifier to make predictions for the input data `X` and compare the predictions with the actual values `y`.


```python
predictions = simple_classifier.predict(X)
np.mean((predictions - y) ** 2)
```




    74401210.603607252



The mean squared error looks quite high. This is likely because our variables (temperature, price of fuel, and unemployment rate) are only weakly correlated with the weekly sales.

There are two more variables in our data that might be more useful for prediction: the `IsHoliday` column and `MarkDown` column. The boxplot below shows that holidays may have some relation with the weekly sales.


```python
sns.pointplot(x='IsHoliday', y='Weekly_Sales', data=walmart);
```


![png](feature_one_hot_files/feature_one_hot_15_0.png)


The different markdown categories seem to correlate with different weekly sale amounts well.


```python
markdowns = ['No Markdown', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
plt.figure(figsize=(7, 5))
sns.pointplot(x='Weekly_Sales', y='MarkDown', data=walmart, order=markdowns);
```


![png](feature_one_hot_files/feature_one_hot_17_0.png)


However, both `IsHoliday` and `MarkDown` columns contain categorical data, not numerical, so we cannot use them as-is for regression.

## The One-Hot Encoding

Fortunately, we can perform a **one-hot encoding** transformation on these categorical variables to convert them into numerical variables. The transformation works as follows: create a new column for every unique value in a categorical variable. The column contains a $1$ if the variable originally had the corresponding value, otherwise the column contains a $0$. For example, the `MarkDown` column below contains the following values:


```python
# HIDDEN
walmart[['MarkDown']]
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
      <th>MarkDown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No Markdown</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>MarkDown2</td>
    </tr>
    <tr>
      <th>141</th>
      <td>MarkDown2</td>
    </tr>
    <tr>
      <th>142</th>
      <td>MarkDown1</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 1 columns</p>
</div>



This variable contains six different unique values: 'No Markdown', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', and 'MarkDown5'. We create one column for each value to get six columns in total. Then, we fill in the columns with zeros and ones according the scheme described above.


```python
# HIDDEN
from sklearn.feature_extraction import DictVectorizer

items = walmart[['MarkDown']].to_dict(orient='records')
encoder = DictVectorizer(sparse=False)
pd.DataFrame(
    data=encoder.fit_transform(items),
    columns=encoder.feature_names_
)
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
      <th>MarkDown=MarkDown1</th>
      <th>MarkDown=MarkDown2</th>
      <th>MarkDown=MarkDown3</th>
      <th>MarkDown=MarkDown4</th>
      <th>MarkDown=MarkDown5</th>
      <th>MarkDown=No Markdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>142</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 6 columns</p>
</div>



Notice that the first value in the data is "No Markdown", and thus only the last column of the first row in the transformed table is marked with $1$. In addition, the last value in the data is "MarkDown1" which results in the first column of row 142 marked as $1$.

Each row of the resulting table will contain a single column containing $1$; the rest will contain $0$. The name "one-hot" reflects the fact that only one column is "hot" (marked with a $1$).

## One-Hot Encoding in Scikit-Learn

To perform one-hot encoding we can use `scikit-learn`'s [`DictVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) class. To use the class, we have to convert our dataframe into a list of dictionaries. The DictVectorizer class automatically one-hot encodes the categorical data (which needs to be strings) and leaves numerical data untouched.


```python
from sklearn.feature_extraction import DictVectorizer

all_columns = ['Temperature', 'Fuel_Price', 'Unemployment', 'IsHoliday',
               'MarkDown']

records = walmart[all_columns].to_dict(orient='records')
encoder = DictVectorizer(sparse=False)
encoded_X = encoder.fit_transform(records)
encoded_X
```




    array([[  2.57,   1.  ,   0.  , ...,   1.  ,  42.31,   8.11],
           [  2.55,   0.  ,   1.  , ...,   1.  ,  38.51,   8.11],
           [  2.51,   1.  ,   0.  , ...,   1.  ,  39.93,   8.11],
           ..., 
           [  3.6 ,   1.  ,   0.  , ...,   0.  ,  62.99,   6.57],
           [  3.59,   1.  ,   0.  , ...,   0.  ,  67.97,   6.57],
           [  3.51,   1.  ,   0.  , ...,   0.  ,  69.16,   6.57]])



To get a better sense of the transformed data, we can display it with the column names:


```python
pd.DataFrame(data=encoded_X, columns=encoder.feature_names_)
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
      <th>Fuel_Price</th>
      <th>IsHoliday=No</th>
      <th>IsHoliday=Yes</th>
      <th>MarkDown=MarkDown1</th>
      <th>...</th>
      <th>MarkDown=MarkDown5</th>
      <th>MarkDown=No Markdown</th>
      <th>Temperature</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.572</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>42.31</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.548</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>38.51</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.514</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>39.93</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>3.601</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>62.99</td>
      <td>6.573</td>
    </tr>
    <tr>
      <th>141</th>
      <td>3.594</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67.97</td>
      <td>6.573</td>
    </tr>
    <tr>
      <th>142</th>
      <td>3.506</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.16</td>
      <td>6.573</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 11 columns</p>
</div>



The numerical variables (fuel price, temperature, and unemployment) are left as numbers. The categorical variables (holidays and markdown) are one-hot encoded. When we use the new matrix of data to fit a linear regression model, we will generate one parameter for each column of the data. Since this data matrix contains eleven columns, the model will have twelve parameters since we fit extra parameter for the intercept term.

## Fitting a Model Using the Transformed Data

We can now use the `encoded_X` variable for linear regression.


```python
clf = LinearRegression()
clf.fit(encoded_X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



As promised, we have eleven parameters for the columns and one intercept parameter.


```python
clf.coef_, clf.intercept_
```




    (array([ 1622.11,    -2.04,     2.04,   962.91,  1805.06, -1748.48,
            -2336.8 ,   215.06,  1102.25,  -330.91,  1205.56]), 29723.135729284979)



We can compare a few of the predictions from both classifiers to see whether there's a large difference between the two.


```python
walmart[['Weekly_Sales']].assign(
    pred_numeric=simple_classifier.predict(X),
    pred_both=clf.predict(encoded_X)
)
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
      <th>Weekly_Sales</th>
      <th>pred_numeric</th>
      <th>pred_both</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24924.50</td>
      <td>30768.878035</td>
      <td>30766.790214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46039.49</td>
      <td>31992.279504</td>
      <td>31989.410395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41595.55</td>
      <td>31465.220158</td>
      <td>31460.280008</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>22764.01</td>
      <td>23492.262649</td>
      <td>24447.348979</td>
    </tr>
    <tr>
      <th>141</th>
      <td>24185.27</td>
      <td>21826.414794</td>
      <td>22788.049554</td>
    </tr>
    <tr>
      <th>142</th>
      <td>27390.81</td>
      <td>21287.928537</td>
      <td>21409.367463</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 3 columns</p>
</div>



It appears that both models make very similar predictions. A scatter plot of both sets of predictions confirms this.


```python
plt.scatter(simple_classifier.predict(X), clf.predict(encoded_X))
plt.title('Predictions using all data vs. numerical features only')
plt.xlabel('Predictions using numerical features')
plt.ylabel('Predictions using all features');
```


![png](feature_one_hot_files/feature_one_hot_36_0.png)


## Model Diagnosis

Why might this be the case? We can examine the parameters that both models learn. The table below shows the weights learned by the classifier that only used numerical variables without one-hot encoding:


```python
# HIDDEN
def clf_params(names, clf):
    weights = (
        np.append(clf.coef_, clf.intercept_)
    )
    return pd.DataFrame(weights, names + ['Intercept'])

clf_params(numerical_columns, simple_classifier)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Temperature</th>
      <td>-332.221180</td>
    </tr>
    <tr>
      <th>Fuel_Price</th>
      <td>1626.625604</td>
    </tr>
    <tr>
      <th>Unemployment</th>
      <td>1356.868319</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>29642.700510</td>
    </tr>
  </tbody>
</table>
</div>



The table below shows the weights learned by the classifier with one-hot encoding.


```python
# HIDDEN
pd.options.display.max_rows = 13
display(clf_params(encoder.feature_names_, clf))
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fuel_Price</th>
      <td>1622.106239</td>
    </tr>
    <tr>
      <th>IsHoliday=No</th>
      <td>-2.041451</td>
    </tr>
    <tr>
      <th>IsHoliday=Yes</th>
      <td>2.041451</td>
    </tr>
    <tr>
      <th>MarkDown=MarkDown1</th>
      <td>962.908849</td>
    </tr>
    <tr>
      <th>MarkDown=MarkDown2</th>
      <td>1805.059613</td>
    </tr>
    <tr>
      <th>MarkDown=MarkDown3</th>
      <td>-1748.475046</td>
    </tr>
    <tr>
      <th>MarkDown=MarkDown4</th>
      <td>-2336.799791</td>
    </tr>
    <tr>
      <th>MarkDown=MarkDown5</th>
      <td>215.060616</td>
    </tr>
    <tr>
      <th>MarkDown=No Markdown</th>
      <td>1102.245760</td>
    </tr>
    <tr>
      <th>Temperature</th>
      <td>-330.912587</td>
    </tr>
    <tr>
      <th>Unemployment</th>
      <td>1205.564331</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>29723.135729</td>
    </tr>
  </tbody>
</table>
</div>


We can see that even when we fit a linear regression model using one-hot encoded columns the weights for fuel price, temperature, and unemployment are very similar to the previous values. All the weights are small in comparison to the intercept term, suggesting that most of the variables are still only slightly correlated with the actual sale amounts. In fact, the model weights for the `IsHoliday` variable are so low that it makes nearly no difference in prediction whether the date was a holiday or not. Although some of the `MarkDown` weights are rather large, many markdown events only appear a few times in the dataset.


```python
walmart['MarkDown'].value_counts()
```




    No Markdown    92
    MarkDown1      25
    MarkDown2      13
    MarkDown5       9
    MarkDown4       2
    MarkDown3       2
    Name: MarkDown, dtype: int64



This suggests that we probably need to collect more data in order for the model to better utilize the effects of markdown events on the sale amounts. (In reality, the dataset shown here is a small subset of a [much larger dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) released by Walmart. It will be a useful exercise to train a model using the entire dataset instead of a small subset.)

## Summary

We have learned to use one-hot encoding, a useful technique for conducting linear regression on categorical data. Although in this particular example the transformation didn't affect our model very much, in practice the technique is used widely when working with categorical data. One-hot encoding also illustrates the general principle of feature engineering—it takes an original data matrix and transforms it into a potentially more useful one.
