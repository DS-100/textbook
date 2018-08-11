
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Getting-Started" data-toc-modified-id="Getting-Started-1">Getting Started</a></span></li><li><span><a href="#Indexes,-Slicing,-and-Sorting" data-toc-modified-id="Indexes,-Slicing,-and-Sorting-2">Indexes, Slicing, and Sorting</a></span><ul class="toc-item"><li><span><a href="#Breaking-the-Problem-Down" data-toc-modified-id="Breaking-the-Problem-Down-2.1">Breaking the Problem Down</a></span></li><li><span><a href="#Slicing-using-.loc" data-toc-modified-id="Slicing-using-.loc-2.2">Slicing using <code>.loc</code></a></span><ul class="toc-item"><li><span><a href="#Slicing-rows-using-a-predicate" data-toc-modified-id="Slicing-rows-using-a-predicate-2.2.1">Slicing rows using a predicate</a></span></li></ul></li><li><span><a href="#Sorting-Rows" data-toc-modified-id="Sorting-Rows-2.3">Sorting Rows</a></span></li></ul></li><li><span><a href="#In-Conclusion" data-toc-modified-id="In-Conclusion-3">In Conclusion</a></span></li></ul></div>


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
```

## Getting Started

In each section of this chapter we will work with the Baby Names dataset from Chapter 1. We will pose a question, break the question down into high-level steps, then translate each step into Python code using `pandas` DataFrames. We begin by importing `pandas`:


```python
# pd is a common shorthand for pandas
import pandas as pd
```

Now we can read in the data using `pd.read_csv` ([docs](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)).


```python
baby = pd.read_csv('babynames.csv')
baby
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>F</td>
      <td>9217</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>F</td>
      <td>3860</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>2587</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1891891</th>
      <td>Verna</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891892</th>
      <td>Winnie</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891893</th>
      <td>Winthrop</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
  </tbody>
</table>
<p>1891894 rows × 4 columns</p>
</div>



Note that for the code above to work, the `babynames.csv` file must be located in the same directory as this notebook. We can check what files are in the current folder by running `ls` in a notebook cell:


```python
ls
```

    babynames.csv                  indexes_slicing_sorting.ipynb


When we use `pandas` to read in data, we get a DataFrame. A DataFrame is a tabular data structure where each column is labeled (in this case 'Name', 'Sex', 'Count', 'Year') and each row is labeled (in this case 0, 1, 2, ..., 1891893). The Table introduced in Data 8, however, only has labeled columns.

The labels of a DataFrame are called the *indexes* of the DataFrame and make many data manipulations easier.

## Indexes, Slicing, and Sorting

Let's use `pandas` to answer the following question:

**What were the five most popular baby names in 2016?**

### Breaking the Problem Down

We can decompose this question into the following simpler table manipulations:

1. Slice out the rows for the year 2016.
2. Sort the rows in descending order by Count.

Now, we can express these steps in `pandas`.

### Slicing using `.loc`

To select subsets of a DataFrame, we use the `.loc` slicing syntax. The first argument is the label of the row and the second is the label of the column:


```python
baby
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>F</td>
      <td>9217</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>F</td>
      <td>3860</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>2587</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1891891</th>
      <td>Verna</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891892</th>
      <td>Winnie</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891893</th>
      <td>Winthrop</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
  </tbody>
</table>
<p>1891894 rows × 4 columns</p>
</div>




```python
baby.loc[1, 'Name'] # Row labeled 1, Column labeled 'Name'
```




    'Anna'



To slice out multiple rows or columns, we can use `:`. Note that `.loc` slicing is inclusive, unlike Python's slicing.


```python
# Get rows 1 through 5, columns Name through Count inclusive
baby.loc[1:5, 'Name':'Count']
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>F</td>
      <td>3860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>2587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elizabeth</td>
      <td>F</td>
      <td>2549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Minnie</td>
      <td>F</td>
      <td>2243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Margaret</td>
      <td>F</td>
      <td>2142</td>
    </tr>
  </tbody>
</table>
</div>



We will often want a single column from a DataFrame:


```python
baby.loc[:, 'Year']
```




    0          1884
    1          1884
    2          1884
               ... 
    1891891    1883
    1891892    1883
    1891893    1883
    Name: Year, Length: 1891894, dtype: int64



Note that when we select a single column, we get a `pandas` Series. A Series is like a one-dimensional NumPy array since we can perform arithmetic on all the elements at once.


```python
baby.loc[:, 'Year'] * 2
```




    0          3768
    1          3768
    2          3768
               ... 
    1891891    3766
    1891892    3766
    1891893    3766
    Name: Year, Length: 1891894, dtype: int64



To select out specific columns, we can pass a list into the `.loc` slice:


```python
# This is a DataFrame again
baby.loc[:, ['Name', 'Year']]
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
      <th>Name</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1891891</th>
      <td>Verna</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891892</th>
      <td>Winnie</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891893</th>
      <td>Winthrop</td>
      <td>1883</td>
    </tr>
  </tbody>
</table>
<p>1891894 rows × 2 columns</p>
</div>



Selecting columns is common, so there's a shorthand.


```python
# Shorthand for baby.loc[:, 'Name']
baby['Name']
```




    0              Mary
    1              Anna
    2              Emma
                 ...   
    1891891       Verna
    1891892      Winnie
    1891893    Winthrop
    Name: Name, Length: 1891894, dtype: object




```python
# Shorthand for baby.loc[:, ['Name', 'Count']]
baby[['Name', 'Count']]
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
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>9217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>3860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>2587</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1891891</th>
      <td>Verna</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1891892</th>
      <td>Winnie</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1891893</th>
      <td>Winthrop</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>1891894 rows × 2 columns</p>
</div>



#### Slicing rows using a predicate

To slice out the rows with year 2016, we will first create a Series containing `True` for each row we want to keep and `False` for each row we want to drop. This is simple because math and boolean operators on Series are applied to each element in the Series.


```python
# Series of years
baby['Year']
```




    0          1884
    1          1884
    2          1884
               ... 
    1891891    1883
    1891892    1883
    1891893    1883
    Name: Year, Length: 1891894, dtype: int64




```python
# Compare each year with 2016
baby['Year'] == 2016
```




    0          False
    1          False
    2          False
               ...  
    1891891    False
    1891892    False
    1891893    False
    Name: Year, Length: 1891894, dtype: bool



Once we have this Series of `True` and `False`, we can pass it into `.loc`.


```python
# We are slicing rows, so the boolean Series goes in the first
# argument to .loc
baby_2016 = baby.loc[baby['Year'] == 2016, :]
baby_2016
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1850880</th>
      <td>Emma</td>
      <td>F</td>
      <td>19414</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1850881</th>
      <td>Olivia</td>
      <td>F</td>
      <td>19246</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1850882</th>
      <td>Ava</td>
      <td>F</td>
      <td>16237</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1883745</th>
      <td>Zyahir</td>
      <td>M</td>
      <td>5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1883746</th>
      <td>Zyel</td>
      <td>M</td>
      <td>5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1883747</th>
      <td>Zylyn</td>
      <td>M</td>
      <td>5</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
<p>32868 rows × 4 columns</p>
</div>



### Sorting Rows

The next step is the sort the rows in descending order by 'Count'. We can use the `sort_values()` function.


```python
sorted_2016 = baby_2016.sort_values('Count', ascending=False)
sorted_2016
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1850880</th>
      <td>Emma</td>
      <td>F</td>
      <td>19414</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1850881</th>
      <td>Olivia</td>
      <td>F</td>
      <td>19246</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1869637</th>
      <td>Noah</td>
      <td>M</td>
      <td>19015</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1868752</th>
      <td>Mikaelyn</td>
      <td>F</td>
      <td>5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1868751</th>
      <td>Miette</td>
      <td>F</td>
      <td>5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1883747</th>
      <td>Zylyn</td>
      <td>M</td>
      <td>5</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
<p>32868 rows × 4 columns</p>
</div>



Finally, we will use `.iloc` to slice out the first five rows of the DataFrame. `.iloc` works like `.loc` but takes in numerical indices instead of labels. It does not include the right endpoint in its slices, like Python's list slicing.


```python
# Get the value in the zeroth row, zeroth column
sorted_2016.iloc[0, 0]
```




    'Emma'




```python
# Get the first five rows
sorted_2016.iloc[0:5]
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
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1850880</th>
      <td>Emma</td>
      <td>F</td>
      <td>19414</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1850881</th>
      <td>Olivia</td>
      <td>F</td>
      <td>19246</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1869637</th>
      <td>Noah</td>
      <td>M</td>
      <td>19015</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1869638</th>
      <td>Liam</td>
      <td>M</td>
      <td>18138</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1850882</th>
      <td>Ava</td>
      <td>F</td>
      <td>16237</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>



## In Conclusion

We now have the five most popular baby names in 2016 and learned to express the following operations in `pandas`:

| Operation | `pandas` |
| --------- | -------  |
| Read a CSV file | `pd.read_csv()` |
| Slicing using labels or indices | `.loc` and `.iloc` |
| Slicing rows using a predicate | Use a boolean-valued Series in `.loc` |
| Sorting rows | `.sort_values()` |
