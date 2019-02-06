

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

## Granularity

The granularity of your data is what each record in your data represents. For example, in the Calls dataset each record represents a single case of a police call.


```python
# HIDDEN
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
      <th>CVLEGEND</th>
      <th>BLKADDR</th>
      <th>EVENTDTTM</th>
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
      <td>BURGLARY - VEHICLE</td>
      <td>2500 LE CONTE AVE</td>
      <td>2017-07-23 06:00:00</td>
      <td>37.876965</td>
      <td>-122.260544</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17038302</td>
      <td>BURGLARY AUTO</td>
      <td>BURGLARY - VEHICLE</td>
      <td>BOWDITCH STREET &amp; CHANNING WAY</td>
      <td>2017-07-02 22:00:00</td>
      <td>37.867209</td>
      <td>-122.256554</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17049346</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>LARCENY</td>
      <td>2900 CHANNING WAY</td>
      <td>2017-08-20 23:20:00</td>
      <td>37.867948</td>
      <td>-122.250664</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17091319</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>LARCENY</td>
      <td>2100 RUSSELL ST</td>
      <td>2017-07-09 04:15:00</td>
      <td>37.856719</td>
      <td>-122.266672</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17044238</td>
      <td>DISTURBANCE</td>
      <td>DISORDERLY CONDUCT</td>
      <td>TELEGRAPH AVENUE &amp; DURANT AVE</td>
      <td>2017-07-30 01:16:00</td>
      <td>37.867816</td>
      <td>-122.258994</td>
      <td>Sunday</td>
    </tr>
  </tbody>
</table>
</div>



In the Stops dataset, each record represents a single incident of a police stop.


```python
# HIDDEN
stops = pd.read_csv('data/stops.csv', parse_dates=[1], infer_datetime_format=True)
stops.head()
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
      <th>Incident Number</th>
      <th>Call Date/Time</th>
      <th>Location</th>
      <th>Incident Type</th>
      <th>Dispositions</th>
      <th>Location - Latitude</th>
      <th>Location - Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-00004825</td>
      <td>2015-01-26 00:10:00</td>
      <td>SAN PABLO AVE / MARIN AVE</td>
      <td>T</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-00004829</td>
      <td>2015-01-26 00:50:00</td>
      <td>SAN PABLO AVE / CHANNING WAY</td>
      <td>T</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-00004831</td>
      <td>2015-01-26 01:03:00</td>
      <td>UNIVERSITY AVE / NINTH ST</td>
      <td>T</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-00004848</td>
      <td>2015-01-26 07:16:00</td>
      <td>2000 BLOCK BERKELEY WAY</td>
      <td>1194</td>
      <td>BM4ICN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-00004849</td>
      <td>2015-01-26 07:43:00</td>
      <td>1700 BLOCK SAN PABLO AVE</td>
      <td>1194</td>
      <td>BM4ICN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



On the other hand, we could have received the Stops data in the following format:


```python
# HIDDEN
(stops
 .groupby(stops['Call Date/Time'].dt.date)
 .size()
 .rename('Num Incidents')
 .to_frame()
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
      <th>Num Incidents</th>
    </tr>
    <tr>
      <th>Call Date/Time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-26</th>
      <td>46</td>
    </tr>
    <tr>
      <th>2015-01-27</th>
      <td>57</td>
    </tr>
    <tr>
      <th>2015-01-28</th>
      <td>56</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-04-28</th>
      <td>82</td>
    </tr>
    <tr>
      <th>2017-04-29</th>
      <td>86</td>
    </tr>
    <tr>
      <th>2017-04-30</th>
      <td>59</td>
    </tr>
  </tbody>
</table>
<p>825 rows × 1 columns</p>
</div>



In this case, each record in the table corresponds to a single date instead of a single incident. We would describe this table as having a coarser granularity than the one above. It's important to know the granularity of your data because it determines what kind of analyses you can perform. Generally speaking, too fine of a granularity is better than too coarse; while we can use grouping and pivoting to change a fine granularity to a coarse one, we have few tools to go from coarse to fine.

## Granularity Checklist

You should have answers to the following questions after looking at the granularity of your datasets. We will answer them for the Calls and Stops datasets.

**What does a record represent?**

In the Calls dataset, each record represents a single case of a police call. In the Stops dataset, each record represents a single incident of a police stop.

**Do all records capture granularity at the same level? (Sometimes a table will contain summary rows.)**

Yes, for both Calls and Stops datasets.

**If the data were aggregated, how was the aggregation performed? Sampling and averaging are are common aggregations.**

No aggregations were performed as far as we can tell for the datasets. We do keep in mind that in both datasets, the location is entered as a block location instead of a specific address.

**What kinds of aggregations can we perform on the data?**

For example, it's often useful to aggregate individual people to demographic groups or individual events to totals across time.

In this case, we can aggregate across various granularities of date or time. For example, we can find the most common hour of day for incidents with aggregation. We might also be able to aggregate across event locations to find the regions of Berkeley with the most incidents.
