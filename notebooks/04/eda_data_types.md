

```python
# HIDDEN
import warnings
# Ignore numpy dtype warnings. These warnings are caused by an interaction
# between numpy and Cython and can be safely ignored.
# Reference: https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```


```python
# HIDDEN
def df_interact(df, nrows=7, ncols=7):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]

    row_arg = (0, len(df), nrows) if len(df) > nrows else fixed(0)
    col_arg = ((0, len(df.columns), ncols)
               if len(df.columns) > ncols else fixed(0))
    
    interact(peek, row=row_arg, col=col_arg)
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))
```


```python
# HIDDEN
babies = pd.read_csv('data/babies23.data', delimiter='\s+')
babies_small = babies[['wt', 'race', 'ed']]
```

## Data Types

We often begin exploratory data analysis by examining the types of data that occur in a table. Although there are multiple ways of categorizing data types, in this book we discuss three broad types of data:

1. **Nominal data**, which represents categories that do not have a natural ordering. For example: names of people, beverage titles, and zip codes.
1. **Ordinal data**, which represents ordered categories. For example: T-shirt sizes (small, medium, large), Likert-scale responses (disagree, neutral, agree), and level of education (high school, college, graduate school).
1. **Numerical data**, which represents amounts or quantities. For example: heights, prices, and distances.

We refer to these types as **statistical data types**, or simply **data types**.

`pandas` assigns each column of a DataFrame a **computational data type** that represents how the data are stored in the computer's memory. It is essential to remember that the statistical data type can differ from the computational data type.

For example, consider the table below which records weights of babies at birth, race of the mother, and educational level of the mother.


```python
babies_small
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
      <th>wt</th>
      <th>race</th>
      <th>ed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>128</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1233</th>
      <td>130</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>125</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>117</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>1236 rows × 3 columns</p>
</div>



Every column of thie DataFrame has a numeric computational data type. In this case, the `int64` type signifies that each column contains integers.


```python
babies_small.dtypes
```




    wt      int64
    race    int64
    ed      int64
    dtype: object



However, it would be foolish to work with all three columns as if they have a numeric statistical data type. In order to understand the dataset's data types, we almost always need to consult the dataset's **data dictionary**. A data dictionary is a document included with the data that describes what each column in the data records. For example, the data dictionary for this dataset states the following:

```
wt -  birth weight in ounces (999 unknown)
race - mother's race 0-5=white 6=mex 7=black 8=asian 9=mixed 99=unknown
ed - mother's education 0= less than 8th grade, 
   1 = 8th -12th grade - did not graduate, 
   2= HS graduate--no other schooling , 3= HS+trade,
   4=HS+some college 5= College graduate, 6&7 Trade school HS unclear, 9=unknown
```

Although the `wt`, `race`, and `ed` columns are stored as integers in `pandas`, the `race` column contains nominal data and `ed` contains ordinal data.

In fact, we must exercise caution even with the `wt` column. Computing the average birth weight by taking the average of the `wt` column will not give an accurate result because unknown values are recorded as `999`. If left as is, the unknown values will cause our average to be higher than it should be.

**The Importance of Data Types**

Data types guide further data analysis by specifying the operations, visualizations, and models we can apply to values in the data. For example, differences between numerical data are meaningful while differences between ordinal data are not. This means that for the `babies_small` DataFrame the average baby birth weight has meaning but not the "average" educational level.

`pandas` will not complain if we attempt to compute the mean of the values in the educational level column:


```python
# Don't use this value in actual data analysis
babies_small['ed'].mean()
```




    2.9215210355987056



This quantity, however, provides little useful information. We could have easily replaced the values in the `ed` column with their string descriptions — for example, we can replace `0`'s with `'less than 8th grade'`, `1`'s with `'8th-12th grade'`, and so on. We would not say that the "average" of these strings contains much value. We would not say the same with the average of the numeric values either.

Although the value differences of ordinal data are not meaningful, the direction of the difference has meaning. For example, we could say a mother with `ed=5` (college graduate) has a greater education level than a mother with `ed=2` (high school graduate).

Nominal data, in comparison, do not provide meaning in the direction of the differences. A mother with `race=6` (Mexican) and a mother with `race=7` (Black) simply have different races.

### Example: Infant Health




```python
babies
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
      <th>id</th>
      <th>pluralty</th>
      <th>outcome</th>
      <th>date</th>
      <th>...</th>
      <th>inc</th>
      <th>smoke</th>
      <th>time</th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>5</td>
      <td>1</td>
      <td>1411</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>5</td>
      <td>1</td>
      <td>1499</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>5</td>
      <td>1</td>
      <td>1576</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <th>1233</th>
      <td>9213</td>
      <td>5</td>
      <td>1</td>
      <td>1672</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>9229</td>
      <td>5</td>
      <td>1</td>
      <td>1680</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>9263</td>
      <td>5</td>
      <td>1</td>
      <td>1668</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1236 rows × 23 columns</p>
</div>




```python
scores = pd.read_csv('data/SFBusinesses/inspections.csv')
scores
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
      <th>business_id</th>
      <th>score</th>
      <th>date</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>94</td>
      <td>20160513</td>
      <td>routine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>94</td>
      <td>20171211</td>
      <td>routine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>98</td>
      <td>20171101</td>
      <td>routine</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14219</th>
      <td>94142</td>
      <td>100</td>
      <td>20171220</td>
      <td>routine</td>
    </tr>
    <tr>
      <th>14220</th>
      <td>94189</td>
      <td>96</td>
      <td>20171130</td>
      <td>routine</td>
    </tr>
    <tr>
      <th>14221</th>
      <td>94231</td>
      <td>85</td>
      <td>20171214</td>
      <td>routine</td>
    </tr>
  </tbody>
</table>
<p>14222 rows × 4 columns</p>
</div>




```python
housing = pd.read_csv('data/SFHousing.csv')
housing
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
      <th>Unnamed: 0</th>
      <th>county</th>
      <th>city</th>
      <th>zip</th>
      <th>...</th>
      <th>lat</th>
      <th>quality</th>
      <th>match</th>
      <th>wk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alameda County</td>
      <td>Alameda</td>
      <td>94501.0</td>
      <td>...</td>
      <td>37.76</td>
      <td>gpsvisualizer</td>
      <td>Exact</td>
      <td>2003-04-21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alameda County</td>
      <td>Alameda</td>
      <td>94501.0</td>
      <td>...</td>
      <td>37.76</td>
      <td>QUALITY_ADDRESS_RANGE_INTERPOLATION</td>
      <td>Exact</td>
      <td>2003-04-21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Alameda County</td>
      <td>Alameda</td>
      <td>94501.0</td>
      <td>...</td>
      <td>37.77</td>
      <td>QUALITY_ADDRESS_RANGE_INTERPOLATION</td>
      <td>Exact</td>
      <td>2003-04-21</td>
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
      <th>281503</th>
      <td>348191</td>
      <td>Sonoma County</td>
      <td>Sonoma</td>
      <td>95476.0</td>
      <td>...</td>
      <td>38.28</td>
      <td>QUALITY_CITY_CENTROID</td>
      <td>Exact</td>
      <td>2006-05-29</td>
    </tr>
    <tr>
      <th>281504</th>
      <td>348192</td>
      <td>Sonoma County</td>
      <td>Windsor</td>
      <td>95492.0</td>
      <td>...</td>
      <td>38.55</td>
      <td>QUALITY_EXACT_PARCEL_CENTROID</td>
      <td>Relaxed; Soundex</td>
      <td>2006-05-29</td>
    </tr>
    <tr>
      <th>281505</th>
      <td>348193</td>
      <td>Sonoma County</td>
      <td>Windsor</td>
      <td>95492.0</td>
      <td>...</td>
      <td>38.54</td>
      <td>QUALITY_CITY_CENTROID</td>
      <td>Exact</td>
      <td>2006-05-29</td>
    </tr>
  </tbody>
</table>
<p>281506 rows × 16 columns</p>
</div>


