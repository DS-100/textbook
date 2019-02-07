

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
babies_small = pd.read_csv('data/babies23.data', delimiter='\s+')[['wt', 'race', 'ed']]
```

## Data Types

We often begin exploratory data analysis by examining the types of data that occur in a table. Although there are multiple ways of categorizing data types, in this book we discuss three broad types of data:

1. **Nominal data**, which represents categories that do not have a natural ordering. For example: political party affiliation (Democrat, Republican, Other), sex (male, female, other), and computer operating system (Windows, MacOS, Linux).
1. **Ordinal data**, which represents ordered categories. For example: T-shirt sizes (small, medium, large), Likert-scale responses (disagree, neutral, agree), and level of education (high school, college, graduate school). Ordinal and nominal data are considered subtypes of categorical data.
1. **Numerical data**, which represents amounts or quantities. For example: heights, prices, and distances.

We refer to these types as **statistical data types**, or simply **data types**.

**Computational vs. Statistical Data Types**

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



Every column of this DataFrame has a numeric computational data type. In this case, the `int64` type signifies that each column contains integers.


```python
babies_small.dtypes
```




    wt      int64
    race    int64
    ed      int64
    dtype: object



However, it does not make sense to work with all three columns as if they have a numeric statistical data type. In order to understand the dataset's data types, we almost always need to consult the dataset's **data dictionary**. A data dictionary is a document included with the data that describes what each column in the data records. For example, the data dictionary for this dataset states the following:

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

Data types guide further data analysis by specifying the operations, visualizations, and models we can meaningfully apply to values in the data. For example, the magnitudes of differences between numerical data are meaningful while the magnitudes of differences between ordinal data are not. This means that for the `babies_small` DataFrame the mean baby birth weight has meaning but not the "mean" educational level.

`pandas` will not complain if we attempt to compute the mean of the values in the educational level column:


```python
# Don't use this value in actual data analysis
babies_small['ed'].mean()
```




    2.9215210355987056



This quantity, however, provides no useful information. We could have easily replaced the values in the `ed` column with their string descriptions — for example, we can replace `0`'s with `'less than 8th grade'`, `1`'s with `'8th-12th grade'`, and so on. We would not say that the "mean" of these strings contains much value. We would not say the same with the mean of the numeric values either.

Although the value differences of ordinal data are not meaningful, the direction of the difference has meaning. For example, we could say a mother with `ed=5` (college graduate) has a greater education level than a mother with `ed=2` (high school graduate).

Nominal data, in comparison, do not provide meaning in the direction of the differences. A mother with `race=6` (Mexican) and a mother with `race=7` (Black) simply have different races.

### Example: Maternal Smoking and Infant Health

The Child Health and Development Studies (CHDS) organization conducts long-term research on how health and disease are passed on between generations (http://www.chdstudies.org/about_us/index.php).

In one notable study, the CHDS collected comprehensive data on all pregnancies between 1960 and 1967 for women using the Kaiser Foundation Health Plan in the San Francisco-East Bay area. Although the CHDS typically requires submitting a request to access its data, a subset of the data on pregnancies is available online through the Stat Labs website ([link][statlabs]). We have downloaded and read the data into a `pandas` DataFrame below.

[statlabs]: https://www.stat.berkeley.edu/users/statlabs/


```python
babies = pd.read_csv('data/babies.data', delimiter='\s+')
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
      <th>bwt</th>
      <th>gestation</th>
      <th>parity</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>smoke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>284</td>
      <td>0</td>
      <td>27</td>
      <td>62</td>
      <td>100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>282</td>
      <td>0</td>
      <td>33</td>
      <td>64</td>
      <td>135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>128</td>
      <td>279</td>
      <td>0</td>
      <td>28</td>
      <td>64</td>
      <td>115</td>
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
    </tr>
    <tr>
      <th>1233</th>
      <td>130</td>
      <td>291</td>
      <td>0</td>
      <td>30</td>
      <td>65</td>
      <td>150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>125</td>
      <td>281</td>
      <td>1</td>
      <td>21</td>
      <td>65</td>
      <td>110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>117</td>
      <td>297</td>
      <td>0</td>
      <td>38</td>
      <td>65</td>
      <td>129</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1236 rows × 7 columns</p>
</div>



To understand the data types of the columns, we refer to the data dictionary. We have copied the data dictionary from the Stat Labs website into this page for convenience:

<table border="2" cellspacing="3" cellpadding="3">
    <thead><tr valign="CENTER">
<td valign="CENTER" align="CENTER">
    <center><strong>&nbsp;Variable &nbsp;</strong></center>
</td>
<td valign="CENTER" align="CENTER">
<center><strong>&nbsp;Description &nbsp;</strong></center>
</td>
</tr></thead>
<tbody>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;bwt
</td>
<td valign="CENTER" align="LEFT">
&nbsp;Birth weight in ounces (999 unknown)
</td>
</tr>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;gestation
</td>
<td valign="CENTER" align="LEFT">
&nbsp;Length of pregnancy in days (999 unknown)
</td>
</tr>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;parity
</td>
<td valign="CENTER" align="LEFT">
&nbsp;0= first born, 9=unknown
</td>
</tr>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;age
</td>
<td valign="CENTER" align="LEFT">
&nbsp;mother's age in years
</td>
</tr>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;height
</td>
<td valign="CENTER" align="LEFT">
&nbsp;mother's height in inches (99 unknown) 
</td>
</tr>
<tr>
<td valign="CENTER" align="LEFT">
&nbsp;weight
</td>
<td valign="CENTER" align="LEFT">
&nbsp;Mother's prepregnancy weight in pounds (999 unknown)
</td>
</tr>

<tr>
<td valign="CENTER" align="LEFT">
&nbsp;smoke
</td>
<td valign="CENTER" align="LEFT">
&nbsp;Smoking status of mother
<br>&nbsp;
 0=not now, 1=yes now, 9=unknown
</td>
</tr>
</tbody></table>

Although the data dictionary does not explicitly specify the data types, we can clearly see that some columns contain nominal data even though all data values appear as numbers in the table. Based on the descriptions of the columns alone, we would likely treat the `parity` and `smoke` columns as nominal data and the remaining columns as numerical.

We hope that the data dictionary provides all necessary information about each column; it is nonetheless useful to double-check our hope by examining the data themselves. For example, it will not always be clear whether a column that records ages contains age values (`21`, `30`, `41`) or age ranges (`21-29`, `30-40`, `41+`).

The `babies` DataFrame appears to exclusively contain integer values:


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
      <th>bwt</th>
      <th>gestation</th>
      <th>parity</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>smoke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>284</td>
      <td>0</td>
      <td>27</td>
      <td>62</td>
      <td>100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>282</td>
      <td>0</td>
      <td>33</td>
      <td>64</td>
      <td>135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>128</td>
      <td>279</td>
      <td>0</td>
      <td>28</td>
      <td>64</td>
      <td>115</td>
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
    </tr>
    <tr>
      <th>1233</th>
      <td>130</td>
      <td>291</td>
      <td>0</td>
      <td>30</td>
      <td>65</td>
      <td>150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>125</td>
      <td>281</td>
      <td>1</td>
      <td>21</td>
      <td>65</td>
      <td>110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>117</td>
      <td>297</td>
      <td>0</td>
      <td>38</td>
      <td>65</td>
      <td>129</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1236 rows × 7 columns</p>
</div>



To peek at a column's values, we can use the `series.value_counts()` method which counts the number of times each value appears.


```python
# Displays the ages (left) and their counts (right)
babies['age'].value_counts()
```




    23    93
    26    90
    24    86
          ..
    45     1
    44     1
    15     1
    Name: age, Length: 31, dtype: int64



We can see that many of the mothers in the dataset were between 24 and 26 years old. In addition, we do not see any fractional ages (e.g. `24.5`), suggesting that the `age` column contains integers.

By default, `pandas` restricts its output to only show a few values at once. To see more possible values, we can ask `pandas` to display more rows, though the output can be verbose if there are many rows to display.


```python
from IPython.display import display

# Within this block, display up to 10 rows. To display more rows,
# change the 10 in the code below to a higher number.
with pd.option_context('display.max_rows', 10):
    display(babies['age'].value_counts())
```


    23    93
    26    90
    24    86
    27    85
    22    79
          ..
    42     4
    99     2
    45     1
    44     1
    15     1
    Name: age, Length: 31, dtype: int64


After double checking the values within the rest of the columns in the DataFrame in a similar way, we label each column with the following data types based on the data dictionary and examination of the data values.

| Variable  | Description                                                | Data Type |
| --------- | ---------------------------------------------------------- | --------- |
| bwt       | Birth weight in ounces (999 unknown)                       | Numerical |
| gestation | Length of pregnancy in days (999 unknown)                  | Numerical |
| parity    | 0= first born, 9=unknown                                   | Nominal   |
| age       | mother's age in years                                      | Numerical |
| height    | mother's height in inches (99 unknown)                     | Numerical |
| weight    | Mother's prepregnancy weight in pounds (999 unknown)       | Numerical |
| smoke     | Smoking status of mother (0=not now, 1=yes now, 9=unknown) | Nominal   |


### Example: San Francisco Restaurant Health Violations

The city of San Francisco, California periodically inspects restaurants for health code violations. At each inspection, the restaurant receives a score from 0 to 100 based on the number and types of violations recorded. The restuarants, scores, and violations are publicly available on the DataSF website ([link](https://data.sfgov.org/Health-and-Social-Services/Restaurant-Scores-LIVES-Standard/pyih-qa8i)); the dataset contains all inspections from January 2016 and onward.

We have loaded a subset of the data into the `scores` DataFrame below.


```python
scores = pd.read_csv('data/SFRestaurants.csv')
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
      <th>business_name</th>
      <th>inspection_score</th>
      <th>violation_description</th>
      <th>risk_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>All stars Donuts</td>
      <td>86.0</td>
      <td>Unclean or degraded floors walls or ceilings</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Soo Fong Restaurant</td>
      <td>92.0</td>
      <td>Wiping cloths not clean or properly stored or ...</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dar Bar Pakistani/Indian Cusine</td>
      <td>86.0</td>
      <td>Moderate risk vermin infestation</td>
      <td>Moderate Risk</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>52795</th>
      <td>USA Power Market</td>
      <td>71.0</td>
      <td>Unclean hands or improper use of gloves</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>52796</th>
      <td>Thai Cottage Restaurant</td>
      <td>81.0</td>
      <td>Inadequate and inaccessible handwashing facili...</td>
      <td>Moderate Risk</td>
    </tr>
    <tr>
      <th>52797</th>
      <td>St Mary's Cathedral/Convention Center</td>
      <td>90.0</td>
      <td>Inadequate and inaccessible handwashing facili...</td>
      <td>Moderate Risk</td>
    </tr>
  </tbody>
</table>
<p>52798 rows × 4 columns</p>
</div>



For this dataset, we are fortunate enough to have a data dictionary that provides some information on the data types:

| Field Name            | Data Type | Description                            |
| --------------------- | --------- | -------------------------------------- |
| business_name         | string    | Common name of the business.           |
| inspection_score      | number    | Calculated inspection score, 0-100     |
| violation_description | string    | One line description of the violation. |
| risk_category         | string    | (No description provided.)             |

However, we see that the data dictionary describes computational data types rather than statistical data types — for example, it uses `string` rather than distinguishing between nominal and ordinal data types. In addition, the `risk_category` column does not have a description.

**Text Data is not Always Nominal Data**

The `violation_description` column provides a description of the violation. However, it is unclear from the data dictionary alone if we should treat the column's value as nominal data. The `violation_description` column might contain **unstructured text**, or free-form text entered in by hand. Unstructured text often originates from open-ended natural language data sources, such as newspaper articles, Google search queries, and free-response survey questions.

We typically do not treat free-form text as nominal data — nominal data typically record a fixed number of categories while unstructured text does not. Thus, we should check the contents of the `violation_description` column to determine whether it contains unstructured text data or nominal data.

If the `violation_description` column contains unstructured text, we would expect that few of its values repeat often because there are many ways to record the same violation: `unclean floors`, `dirty floors`, `floor needs cleaning`, and so on. On the other hand, if the `violation_description` column contains data selected from a pre-specified list, many of the values should repeat.

We indeed find that the column contains many repeat entries:


```python
with pd.option_context('display.max_rows', 14):
    display(scores['violation_description'].value_counts())
```


    Unclean or degraded floors walls or ceilings                          3668
    Unapproved or unmaintained equipment or utensils                      2704
    Inadequate and inaccessible handwashing facilities                    2653
    Moderate risk food holding temperature                                2588
    Inadequately cleaned or sanitized food contact surfaces               2467
    Wiping cloths not clean or properly stored or inadequate sanitizer    2121
    Foods not protected from contamination                                1929
                                                                          ... 
    Discharge from employee nose mouth or eye                                6
    Noncompliance with Gulf Coast oyster regulation                          5
    Mobile food facility stored in unapproved location                       4
    Mobile food facility with unapproved operating conditions                3
    Unreported or unrestricted ill employee with communicable disease        1
    Noncompliance with Cottage Food Operation                                1
    Mobile food facility HCD insignia unavailable                            1
    Name: violation_description, Length: 67, dtype: int64


This leads us to believe that the values in the `violation_description` column were selected from a list of possible violations. Thus, we treat the `violation_description` column as if it contains nominal data.

**Investigating a Missing Description**

Although the `risk_category` column was not described in the data dictionary, we can examine the column's contents to infer its meaning. First, we find that the column only has three unique values:


```python
scores['risk_category'].value_counts()
```




    Low Risk         19694
    Moderate Risk    14712
    High Risk         5686
    Name: risk_category, dtype: int64



We might sense that these values describe the severity of the violation. We can check this intuition by examining violations for each value in the `risk_category` column.


```python
def risk_counts(risk):
    return (scores.loc[scores['risk_category'] == risk,
                       'violation_description']
            .value_counts().head())
```


```python
risk_counts('High Risk')
```




    High risk food holding temperature             1619
    Unclean or unsanitary food contact surfaces    1197
    Improper cooling methods                        823
    Unclean hands or improper use of gloves         755
    High risk vermin infestation                    712
    Name: violation_description, dtype: int64




```python
risk_counts('Moderate Risk')
```




    Inadequate and inaccessible handwashing facilities         2653
    Moderate risk food holding temperature                     2588
    Inadequately cleaned or sanitized food contact surfaces    2467
    Foods not protected from contamination                     1929
    Moderate risk vermin infestation                           1814
    Name: violation_description, dtype: int64




```python
risk_counts('Low Risk')
```




    Unclean or degraded floors walls or ceilings                          3668
    Unapproved or unmaintained equipment or utensils                      2704
    Wiping cloths not clean or properly stored or inadequate sanitizer    2121
    Improper food storage                                                 1817
    Unclean nonfood contact surfaces                                      1440
    Name: violation_description, dtype: int64



At a cursory glance, the violation categories are divided into each risk level. In addition, the `High Risk` violations seem more likely to cause illness than the `Low Risk` violations. Presumably a restaurant with `High risk vermin infestation` would be less sanitary than a restaurant with `Unclean nonfood contact surfaces`.

Because of this, we decide that the `risk_category` column contains ordinal data describing the risk level of each violation (`Low Risk` < `Moderate Risk` < `High Risk`).

**A Revised Data Dictionary**

After investigating the remaining columns of the `scores` DataFrame, we arrive at the following revised data dictionary that we can use for further analysis:

| Field Name            | Data Type | Description                                                        |
| --------------------- | --------- | ------------------------------------------------------------------ |
| business_name         | nominal   | Common name of the business.                                       |
| inspection_score      | numerical | Calculated inspection score, 0-100                                 |
| violation_description | nominal   | One line description of the violation.                             |
| risk_category         | ordinal   | Risk level of the violation (Low Risk < Moderate Risk < High Risk) |


## Summary

We introduced the nominal, ordinal, and numerical data types and their importance for data science. When presented with a dataset, consult the data dictionary and the data itself to determine the data types for each column. Ensure that computational data types are not confused with statistical data types.
