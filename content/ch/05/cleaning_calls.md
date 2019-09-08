

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

## Investigating Berkeley Police Data

We will use the Berkeley Police Department's publicly available datasets to demonstrate data cleaning techniques. We have downloaded the [Calls for Service dataset][calls] and [Stops dataset][stops].

We can use the `ls` shell command with the `-lh` flags to see more details about the files:

[calls]: https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5
[stops]: https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Stop-Data/6e9j-pj9p


```python
!ls -lh data/
```

    total 13936
    -rw-r--r--@ 1 sam  staff   979K Aug 29 14:41 Berkeley_PD_-_Calls_for_Service.csv
    -rw-r--r--@ 1 sam  staff    81B Aug 29 14:28 cvdow.csv
    -rw-r--r--@ 1 sam  staff   5.8M Aug 29 14:41 stops.json


The command above shows the data files and their file sizes. This is especially useful because we now know the files are small enough to load into memory. As a rule of thumb, it is usually safe to load a file into memory that is around one fourth of the total memory capacity of the computer. For example, if a computer has 4GB of RAM we should be able to load a 1GB CSV file in `pandas`. To handle larger datasets we will need additional computational tools that we will cover later in this book.

Notice the use of the exclamation point before `ls`. This tells Jupyter that the next line of code is a shell command, not a Python expression. We can run any available shell command in Jupyter using `!`:


```python
# The `wc` shell command shows us how many lines each file has.
# We can see that the `stops.json` file has the most lines (29852).
!wc -l data/*
```

       16497 data/Berkeley_PD_-_Calls_for_Service.csv
           8 data/cvdow.csv
       29852 data/stops.json
       46357 total


### Understanding the Data Generation

We will state important questions you should ask of all datasets before data cleaning or processing. These questions are related to how the data were generated, so data cleaning will usually **not** be able to resolve issues that arise here.

**What do the data contain?** The website for the Calls for Service data states that the dataset describes "crime incidents (not criminal reports) within the last 180 days". Further reading reveals that "not all calls for police service are included (e.g. Animal Bite)".

The website for the Stops data states that the dataset contains data on all "vehicle detentions (including bicycles) and pedestrian detentions (up to five persons)" since January 26, 2015.

**Are the data a census?** This depends on our population of interest. For example, if we are interested in calls for service within the last 180 days for crime incidents then the Calls dataset is a census. However, if we are interested in calls for service within the last 10 years the dataset is clearly not a census. We can make similar statements about the Stops dataset since the data collection started on January 26, 2015.

**If the data form a sample, is it a probability sample?** If we are investigating a period of time that the data do not have entries for, the data do not form a probability sample since there is no randomness involved in the data collection process — we have all data for certain time periods but no data for others.

**What limitations will this data have on our conclusions?** Although we will ask this question at each step of our data processing, we can already see that our data impose important limitations. The most important limitation is that we cannot make unbiased estimations for time periods not covered by our datasets.

## Cleaning The Calls Dataset

Let's now clean the Calls dataset. The `head` shell command prints the first five lines of the file.


```python
!head data/Berkeley_PD_-_Calls_for_Service.csv
```

    CASENO,OFFENSE,EVENTDT,EVENTTM,CVLEGEND,CVDOW,InDbDate,Block_Location,BLKADDR,City,State
    17091420,BURGLARY AUTO,07/23/2017 12:00:00 AM,06:00,BURGLARY - VEHICLE,0,08/29/2017 08:28:05 AM,"2500 LE CONTE AVE
    Berkeley, CA
    (37.876965, -122.260544)",2500 LE CONTE AVE,Berkeley,CA
    17020462,THEFT FROM PERSON,04/13/2017 12:00:00 AM,08:45,LARCENY,4,08/29/2017 08:28:00 AM,"2200 SHATTUCK AVE
    Berkeley, CA
    (37.869363, -122.268028)",2200 SHATTUCK AVE,Berkeley,CA
    17050275,BURGLARY AUTO,08/24/2017 12:00:00 AM,18:30,BURGLARY - VEHICLE,4,08/29/2017 08:28:06 AM,"200 UNIVERSITY AVE
    Berkeley, CA
    (37.865491, -122.310065)",200 UNIVERSITY AVE,Berkeley,CA


It appears to be a comma-separated values (CSV) file, though it's hard to tell whether the entire file is formatted properly. We can use `pd.read_csv` to read in the file as a DataFrame. If `pd.read_csv` errors, we will have to dig deeper and manually resolve formatting issues. Fortunately, `pd.read_csv` successfully returns a DataFrame:


```python
calls = pd.read_csv('data/Berkeley_PD_-_Calls_for_Service.csv')
calls
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
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
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
      <td>2500 LE CONTE AVE\nBerkeley, CA\n(37.876965, -...</td>
      <td>2500 LE CONTE AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17020462</td>
      <td>THEFT FROM PERSON</td>
      <td>04/13/2017 12:00:00 AM</td>
      <td>08:45</td>
      <td>...</td>
      <td>2200 SHATTUCK AVE\nBerkeley, CA\n(37.869363, -...</td>
      <td>2200 SHATTUCK AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17050275</td>
      <td>BURGLARY AUTO</td>
      <td>08/24/2017 12:00:00 AM</td>
      <td>18:30</td>
      <td>...</td>
      <td>200 UNIVERSITY AVE\nBerkeley, CA\n(37.865491, ...</td>
      <td>200 UNIVERSITY AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
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
      <th>5505</th>
      <td>17018126</td>
      <td>DISTURBANCE</td>
      <td>04/01/2017 12:00:00 AM</td>
      <td>12:22</td>
      <td>...</td>
      <td>1600 FAIRVIEW ST\nBerkeley, CA\n(37.850001, -1...</td>
      <td>1600 FAIRVIEW ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5506</th>
      <td>17090665</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2017 12:00:00 AM</td>
      <td>12:00</td>
      <td>...</td>
      <td>2000 DELAWARE ST\nBerkeley, CA\n(37.874489, -1...</td>
      <td>2000 DELAWARE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5507</th>
      <td>17049700</td>
      <td>SEXUAL ASSAULT MISD.</td>
      <td>08/22/2017 12:00:00 AM</td>
      <td>20:02</td>
      <td>...</td>
      <td>2400 TELEGRAPH AVE\nBerkeley, CA\n(37.866761, ...</td>
      <td>2400 TELEGRAPH AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
<p>5508 rows × 11 columns</p>
</div>



We can define a function to show different slices of the data and then interact with it:


```python
def df_interact(df):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + 5, col:col + 6]
    interact(peek, row=(0, len(df), 5), col=(0, len(df.columns) - 6))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

df_interact(calls)
```


    A Jupyter Widget


    (5508 rows, 11 columns) total


Based on the output above, the resulting DataFrame looks reasonably well-formed since the columns are properly named and the data in each column seems to be entered consistently. What data does each column contain? We can look at the dataset website:

| Column         | Description                            | Type        |
| ------         | -----------                            | ----        |
| CASENO         | Case Number                            | Number      |
| OFFENSE        | Offense Type                           | Plain Text  |
| EVENTDT        | Date Event Occurred                    | Date & Time |
| EVENTTM        | Time Event Occurred                    | Plain Text  |
| CVLEGEND       | Description of Event                   | Plain Text  |
| CVDOW          | Day of Week Event Occurred             | Number      |
| InDbDate       | Date dataset was updated in the portal | Date & Time |
| Block_Location | Block level address of event           | Location    |
| BLKADDR        |                                        | Plain Text  |
| City           |                                        | Plain Text  |
| State          |                                        | Plain Text  |

On the surface the data looks easy to work with. However, before starting data analysis we must answer the following questions:

1. **Are there missing values in the dataset?** This question is important because missing values can represent many different things. For example, missing addresses could mean that locations were removed to protect anonymity, or that some respondents chose not to answer a survey question, or that a recording device broke.
1. **Are there any missing values that were filled in (e.g. a 999 for unknown age or 12:00am for unknown date)?** These will clearly impact analysis if we ignore them.
1. **Which parts of the data were entered by a human?** As we will soon see, human-entered data is filled with inconsistencies and mispellings.

Although there are plenty more checks to go through, these three will suffice for many cases. See the [Quartz bad data guide](https://github.com/Quartz/bad-data-guide) for a more complete list of checks.

### Are there missing values?

This is a simple check in `pandas`:


```python
# True if row contains at least one null value
null_rows = calls.isnull().any(axis=1)
calls[null_rows]
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
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116</th>
      <td>17014831</td>
      <td>BURGLARY AUTO</td>
      <td>03/16/2017 12:00:00 AM</td>
      <td>22:00</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>478</th>
      <td>17042511</td>
      <td>BURGLARY AUTO</td>
      <td>07/20/2017 12:00:00 AM</td>
      <td>16:00</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>486</th>
      <td>17022572</td>
      <td>VEHICLE STOLEN</td>
      <td>04/22/2017 12:00:00 AM</td>
      <td>21:00</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
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
      <th>4945</th>
      <td>17091287</td>
      <td>VANDALISM</td>
      <td>07/01/2017 12:00:00 AM</td>
      <td>08:00</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>4947</th>
      <td>17038382</td>
      <td>BURGLARY RESIDENTIAL</td>
      <td>06/30/2017 12:00:00 AM</td>
      <td>15:00</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5167</th>
      <td>17091632</td>
      <td>VANDALISM</td>
      <td>08/15/2017 12:00:00 AM</td>
      <td>23:30</td>
      <td>...</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
<p>27 rows × 11 columns</p>
</div>



It looks like 27 calls didn't have a recorded address in BLKADDR. Unfortunately, the data description isn't very clear on how the locations were recorded. We know that all of these calls were made for events in Berkeley, so we can likely assume that the addresses for these calls were originally somewhere in Berkeley.

### Are there any missing values that were filled in?

From the missing value check above we can see that the Block_Location column has Berkeley, CA recorded if the location was missing.

In addition, an inspection of the `calls` table shows us that the EVENTDT column has the correct dates but records 12am for all of its times. Instead, the times are in the EVENTTM column.


```python
# Show the first 7 rows of the table again for reference
calls.head(7)
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
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
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
      <td>2500 LE CONTE AVE\nBerkeley, CA\n(37.876965, -...</td>
      <td>2500 LE CONTE AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17020462</td>
      <td>THEFT FROM PERSON</td>
      <td>04/13/2017 12:00:00 AM</td>
      <td>08:45</td>
      <td>...</td>
      <td>2200 SHATTUCK AVE\nBerkeley, CA\n(37.869363, -...</td>
      <td>2200 SHATTUCK AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17050275</td>
      <td>BURGLARY AUTO</td>
      <td>08/24/2017 12:00:00 AM</td>
      <td>18:30</td>
      <td>...</td>
      <td>200 UNIVERSITY AVE\nBerkeley, CA\n(37.865491, ...</td>
      <td>200 UNIVERSITY AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17019145</td>
      <td>GUN/WEAPON</td>
      <td>04/06/2017 12:00:00 AM</td>
      <td>17:30</td>
      <td>...</td>
      <td>1900 SEVENTH ST\nBerkeley, CA\n(37.869318, -12...</td>
      <td>1900 SEVENTH ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17044993</td>
      <td>VEHICLE STOLEN</td>
      <td>08/01/2017 12:00:00 AM</td>
      <td>18:00</td>
      <td>...</td>
      <td>100 PARKSIDE DR\nBerkeley, CA\n(37.854247, -12...</td>
      <td>100 PARKSIDE DR</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17037319</td>
      <td>BURGLARY RESIDENTIAL</td>
      <td>06/28/2017 12:00:00 AM</td>
      <td>12:00</td>
      <td>...</td>
      <td>1500 PRINCE ST\nBerkeley, CA\n(37.851503, -122...</td>
      <td>1500 PRINCE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>17030791</td>
      <td>BURGLARY RESIDENTIAL</td>
      <td>05/30/2017 12:00:00 AM</td>
      <td>08:45</td>
      <td>...</td>
      <td>300 MENLO PL\nBerkeley, CA\n</td>
      <td>300 MENLO PL</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 11 columns</p>
</div>



As a data cleaning step, we want to merge the EVENTDT and EVENTTM columns to record both date and time in one field. If we define a function that takes in a DF and returns a new DF, we can later use [`pd.pipe`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pipe.html) to apply all transformations in one go.


```python
def combine_event_datetimes(calls):
    combined = pd.to_datetime(
        # Combine date and time strings
        calls['EVENTDT'].str[:10] + ' ' + calls['EVENTTM'],
        infer_datetime_format=True,
    )
    return calls.assign(EVENTDTTM=combined)

# To peek at the result without mutating the calls DF:
calls.pipe(combine_event_datetimes).head(2)
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
      <th>City</th>
      <th>State</th>
      <th>EVENTDTTM</th>
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
      <td>Berkeley</td>
      <td>CA</td>
      <td>2017-07-23 06:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17020462</td>
      <td>THEFT FROM PERSON</td>
      <td>04/13/2017 12:00:00 AM</td>
      <td>08:45</td>
      <td>...</td>
      <td>2200 SHATTUCK AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>2017-04-13 08:45:00</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 12 columns</p>
</div>



### Which parts of the data were entered by a human?

It looks like most of the data columns are machine-recorded, including the date, time, day of week, and location of the event.

In addition, the OFFENSE and CVLEGEND columns appear to contain consistent values. We can check the unique values in each column to see if anything was misspelled:


```python
calls['OFFENSE'].unique()
```




    array(['BURGLARY AUTO', 'THEFT FROM PERSON', 'GUN/WEAPON',
           'VEHICLE STOLEN', 'BURGLARY RESIDENTIAL', 'VANDALISM',
           'DISTURBANCE', 'THEFT MISD. (UNDER $950)', 'THEFT FROM AUTO',
           'DOMESTIC VIOLENCE', 'THEFT FELONY (OVER $950)', 'ALCOHOL OFFENSE',
           'MISSING JUVENILE', 'ROBBERY', 'IDENTITY THEFT',
           'ASSAULT/BATTERY MISD.', '2ND RESPONSE', 'BRANDISHING',
           'MISSING ADULT', 'NARCOTICS', 'FRAUD/FORGERY',
           'ASSAULT/BATTERY FEL.', 'BURGLARY COMMERCIAL', 'MUNICIPAL CODE',
           'ARSON', 'SEXUAL ASSAULT FEL.', 'VEHICLE RECOVERED',
           'SEXUAL ASSAULT MISD.', 'KIDNAPPING', 'VICE', 'HOMICIDE'], dtype=object)




```python
calls['CVLEGEND'].unique()
```




    array(['BURGLARY - VEHICLE', 'LARCENY', 'WEAPONS OFFENSE',
           'MOTOR VEHICLE THEFT', 'BURGLARY - RESIDENTIAL', 'VANDALISM',
           'DISORDERLY CONDUCT', 'LARCENY - FROM VEHICLE', 'FAMILY OFFENSE',
           'LIQUOR LAW VIOLATION', 'MISSING PERSON', 'ROBBERY', 'FRAUD',
           'ASSAULT', 'NOISE VIOLATION', 'DRUG VIOLATION',
           'BURGLARY - COMMERCIAL', 'ALL OTHER OFFENSES', 'ARSON', 'SEX CRIME',
           'RECOVERED VEHICLE', 'KIDNAPPING', 'HOMICIDE'], dtype=object)



Since each value in these columns appears to be spelled correctly, we won't have to perform any corrections on these columns.

We also check the BLKADDR column for inconsistencies and find that sometimes an address is recorded (e.g. 2500 LE CONTE AVE) but other times a cross street is recorded (e.g. ALLSTON WAY & FIFTH ST). This suggests that a human entered this data in and this column will be difficult to use for analysis. Fortunately we can use the latitude and longitude of the event instead of the street address.


```python
calls['BLKADDR'][[0, 5001]]
```




    0            2500 LE CONTE AVE
    5001    ALLSTON WAY & FIFTH ST
    Name: BLKADDR, dtype: object



### Final Touchups

This dataset seems almost ready for analysis. The Block_Location column seems to contain strings that record address, latitude, and longitude. We will want to separate the latitude and longitude for easier use.


```python
def split_lat_lon(calls):
    return calls.join(
        calls['Block_Location']
        # Get coords from string
        .str.split('\n').str[2]
        # Remove parens from coords
        .str[1:-1]
        # Split latitude and longitude
        .str.split(', ', expand=True)
        .rename(columns={0: 'Latitude', 1: 'Longitude'})
    )

calls.pipe(split_lat_lon).head(2)
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
      <th>City</th>
      <th>State</th>
      <th>Latitude</th>
      <th>Longitude</th>
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
      <td>Berkeley</td>
      <td>CA</td>
      <td>37.876965</td>
      <td>-122.260544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17020462</td>
      <td>THEFT FROM PERSON</td>
      <td>04/13/2017 12:00:00 AM</td>
      <td>08:45</td>
      <td>...</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>37.869363</td>
      <td>-122.268028</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 13 columns</p>
</div>



Then, we can match the day of week number with its weekday:


```python
# This DF contains the day for each number in CVDOW
day_of_week = pd.read_csv('data/cvdow.csv')
day_of_week
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
      <th>CVDOW</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
</div>




```python
def match_weekday(calls):
    return calls.merge(day_of_week, on='CVDOW')
calls.pipe(match_weekday).head(2)
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
      <th>City</th>
      <th>State</th>
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
      <td>Berkeley</td>
      <td>CA</td>
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
      <td>Berkeley</td>
      <td>CA</td>
      <td>Sunday</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 12 columns</p>
</div>



We'll drop columns we no longer need:


```python
def drop_unneeded_cols(calls):
    return calls.drop(columns=['CVDOW', 'InDbDate', 'Block_Location', 'City',
                               'State', 'EVENTDT', 'EVENTTM'])
```

Finally, we'll pipe the `calls` DF through all the functions we've defined:


```python
calls_final = (calls.pipe(combine_event_datetimes)
               .pipe(split_lat_lon)
               .pipe(match_weekday)
               .pipe(drop_unneeded_cols))
df_interact(calls_final)
```


    A Jupyter Widget


    (5508 rows, 8 columns) total


The Calls dataset is now ready for further data analysis. In the next section, we will clean the Stops dataset.


```python
# HIDDEN
# Save data to CSV for other chapters
# calls_final.to_csv('../ch5/data/calls.csv', index=False)
```
