

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

## Grouping and Pivoting

In this section, we will answer the question:

**What were the most popular male and female names in each year?**

Here's the Baby Names dataset once again:


```python
baby = pd.read_csv('babynames.csv')
baby.head()
# the .head() method outputs the first five rows of the DataFrame
```

### Breaking the Problem Down

We should first notice that the question in the previous section has similarities to this one; the question in the previous section restricts names to babies born in 2016 whereas this question asks for names in all years.

We once again decompose this problem into simpler table manipulations.

1. Group the `baby` DataFrame by 'Year' and 'Sex'.
2. For each group, compute the most popular name.

Recognizing which operation is needed for each problem is sometimes tricky. Usually, a convoluted series of steps will signal to you that there might be a simpler way to express what you want. If we didn't immediately recognize that we needed to group, for example, we might write steps like the following:

1. Loop through each unique year.
2. For each year, loop through each unique sex.
3. For each unique year and sex, find the most common name.

There is almost always a better alternative to looping over a `pandas` DataFrame. **In particular, looping over unique values of a DataFrame should usually be replaced with a group.**

### Grouping

To group in `pandas`. we use the `.groupby()` method.


```python
baby.groupby('Year')
```

`.groupby()` returns a strange-looking `DataFrameGroupBy` object. We can call `.agg()` on this object with an aggregation function in order to get a familiar output:


```python
# The aggregation function takes in a series of values for each group
# and outputs a single value
def length(series):
    return len(series)

# Count up number of values for each year. This is equivalent to
# counting the number of rows where each year appears.
baby.groupby('Year').agg(length)
```

You might notice that the `length` function simply calls the `len` function, so we can simplify the code above.


```python
baby.groupby('Year').agg(len)
```

The aggregation is applied to each column of the DataFrame, producing redundant information. We can restrict the output columns by slicing before grouping.


```python
year_rows = baby[['Year', 'Count']].groupby('Year').agg(len)
year_rows

# A further shorthand to accomplish the same result:
#
# year_counts = baby[['Year', 'Count']].groupby('Year').count()
#
# pandas has shorthands for common aggregation functions, including
# count, sum, and mean.
```

Note that the index of the resulting DataFrame now contains the unique years, so we can slice subsets of years using `.loc` as before:


```python
# Every twentieth year starting at 1880
year_rows.loc[1880:2016:20, :]
```

### Grouping on Multiple Columns

As we've seen in Data 8, we can group on multiple columns to get groups based on unique pairs of values. To do this, pass in a list of column labels into `.groupby()`.


```python
grouped_counts = baby.groupby(['Year', 'Sex']).sum()
grouped_counts
```

The code above computes the total number of babies born for each year and sex. Let's now use grouping by muliple columns to compute the most popular names for each year and sex. Since the data are already sorted in descending order of Count for each year and sex, we can define an aggregation function that returns the first value in each series. (If the data weren't sorted, we can call `sort_values()` first.)


```python
# The most popular name is simply the first one that appears in the series
def most_popular(series):
    return series.iloc[0]

baby_pop = baby.groupby(['Year', 'Sex']).agg(most_popular)
baby_pop
```

Notice that grouping by multiple columns results in multiple labels for each row. This is called a "multilevel index" and is tricky to work with. The important thing to know is that `.loc` takes in a tuple for the row index instead of a single value:


```python
baby_pop.loc[(2000, 'F'), 'Name']
```

But `.iloc` behaves the same as usual since it uses indices instead of labels:


```python
baby_pop.iloc[10:15, :]
```

### Pivoting

**If you group by two columns, you can often use pivot to present your data in a more convenient format.** Using a pivot lets you use one set of grouped labels as the columns of the resulting table.

To pivot, use the `pd.pivot_table()` function.


```python
pd.pivot_table(baby,
               index='Year',         # Index for rows
               columns='Sex',        # Columns
               values='Name',        # Values in table
               aggfunc=most_popular) # Aggregation function
```

Compare this result to the `baby_pop` table that we computed using `.groupby()`. We can see that the `Sex` index in `baby_pop` became the columns of the pivot table.


```python
baby_pop
```

## In Conclusion

We now have the most popular baby names for each sex and year in our dataset and learned to express the following operations in `pandas`:

| Operation | `pandas` |
| --------- | -------  |
| Group | `df.groupby(label)` |
| Group by multiple columns | `df.groupby([label1, label2])` |
| Group and aggregate | `df.groupby(label).agg(func)` |
| Pivot | `pd.pivot_table()` |
