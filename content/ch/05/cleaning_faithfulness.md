

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

## Faithfulness

We describe a dataset as "faithful" if we believe it accurately captures reality. Typically, untrustworthy datasets contain:

**Unrealistic or incorrect values**

For example, dates in the future, locations that don't exist, negative counts, or large outliers.

**Violations of obvious dependencies**

For example, age and birthday for individuals don't match.

**Hand-entered data**

As we have seen, these are typically filled with spelling errors and inconsistencies.

**Clear signs of data falsification**

For example, repeated names, fake looking email addresses, or repeated use of uncommon names or fields.

Notice the many similarities to data cleaning. As we have mentioned, we often go back and forth between data cleaning and EDA, especially when determining data faithfulness. For example, visualizations often help us identify strange entries in the data.


```python
calls = pd.read_csv('data/calls.csv')
calls.head()
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
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>...</th>
      <th>BLKADDR</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17091420</td>
      <td>BURGLARY AUTO</td>
      <td>07/23/2017 12:00:00 AM</td>
      <td>06:00</td>
      <td>...</td>
      <td>2500 LE CONTE AVE</td>
      <td>37.876965</td>
      <td>-122.260544</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17038302</td>
      <td>BURGLARY AUTO</td>
      <td>07/02/2017 12:00:00 AM</td>
      <td>22:00</td>
      <td>...</td>
      <td>BOWDITCH STREET &amp; CHANNING WAY</td>
      <td>37.867209</td>
      <td>-122.256554</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17049346</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>08/20/2017 12:00:00 AM</td>
      <td>23:20</td>
      <td>...</td>
      <td>2900 CHANNING WAY</td>
      <td>37.867948</td>
      <td>-122.250664</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17091319</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>07/09/2017 12:00:00 AM</td>
      <td>04:15</td>
      <td>...</td>
      <td>2100 RUSSELL ST</td>
      <td>37.856719</td>
      <td>-122.266672</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17044238</td>
      <td>DISTURBANCE</td>
      <td>07/30/2017 12:00:00 AM</td>
      <td>01:16</td>
      <td>...</td>
      <td>TELEGRAPH AVENUE &amp; DURANT AVE</td>
      <td>37.867816</td>
      <td>-122.258994</td>
      <td>Sunday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9 columns</p>
</div>




```python
calls['CASENO'].plot.hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1ebb2898>




![png](cleaning_faithfulness_files/cleaning_faithfulness_3_1.png)


Notice the unexpected clusters at 17030000 and 17090000. By plotting the distribution of case numbers, we can quickly see anomalies in the data. In this case, we might guess that two different teams of police use different sets of case numbers for their calls.

Exploring the data often reveals anomalies; if fixable, we can then apply data cleaning techniques.
