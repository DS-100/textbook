

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

Note that for the code above to work, the `babynames.csv` file must be located in the same directory as this notebook. We can check what files are in the current folder by running `ls` in a notebook cell:


```python
ls
```

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


```python
baby.loc[1, 'Name'] # Row labeled 1, Column labeled 'Name'
```

To slice out multiple rows or columns, we can use `:`. Note that `.loc` slicing is inclusive, unlike Python's slicing.


```python
# Get rows 1 through 5, columns Name through Count inclusive
baby.loc[1:5, 'Name':'Count']
```

We will often want a single column from a DataFrame:


```python
baby.loc[:, 'Year']
```

Note that when we select a single column, we get a `pandas` Series. A Series is like a one-dimensional NumPy array since we can perform arithmetic on all the elements at once.


```python
baby.loc[:, 'Year'] * 2
```

To select out specific columns, we can pass a list into the `.loc` slice:


```python
# This is a DataFrame again
baby.loc[:, ['Name', 'Year']]
```

Selecting columns is common, so there's a shorthand.


```python
# Shorthand for baby.loc[:, 'Name']
baby['Name']
```


```python
# Shorthand for baby.loc[:, ['Name', 'Count']]
baby[['Name', 'Count']]
```

#### Slicing rows using a predicate

To slice out the rows with year 2016, we will first create a Series containing `True` for each row we want to keep and `False` for each row we want to drop. This is simple because math and boolean operators on Series are applied to each element in the Series.


```python
# Series of years
baby['Year']
```


```python
# Compare each year with 2016
baby['Year'] == 2016
```

Once we have this Series of `True` and `False`, we can pass it into `.loc`.


```python
# We are slicing rows, so the boolean Series goes in the first
# argument to .loc
baby_2016 = baby.loc[baby['Year'] == 2016, :]
baby_2016
```

### Sorting Rows

The next step is the sort the rows in descending order by 'Count'. We can use the `sort_values()` function.


```python
sorted_2016 = baby_2016.sort_values('Count', ascending=False)
sorted_2016
```

Finally, we will use `.iloc` to slice out the first five rows of the DataFrame. `.iloc` works like `.loc` but takes in numerical indices instead of labels. It does not include the right endpoint in its slices, like Python's list slicing.


```python
# Get the value in the zeroth row, zeroth column
sorted_2016.iloc[0, 0]
```


```python
# Get the first five rows
sorted_2016.iloc[0:5]
```

## In Conclusion

We now have the five most popular baby names in 2016 and learned to express the following operations in `pandas`:

| Operation | `pandas` |
| --------- | -------  |
| Read a CSV file | `pd.read_csv()` |
| Slicing using labels or indices | `.loc` and `.iloc` |
| Slicing rows using a predicate | Use a boolean-valued Series in `.loc` |
| Sorting rows | `.sort_values()` |
