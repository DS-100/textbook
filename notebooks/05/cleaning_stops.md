

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
def df_interact(df):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0):
        return df[row:row + 5]
    interact(peek, row=(0, len(df), 5))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))
```

## Cleaning The Stops Dataset

The Stops dataset ([webpage](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Stop-Data/6e9j-pj9p)) records police stops of pedestrians and vehicles. Let's prepare it for further analysis.

We can use the `head` command to display the first few lines of the file.


```python
!head data/stops.json
```

    {
      "meta" : {
        "view" : {
          "id" : "6e9j-pj9p",
          "name" : "Berkeley PD - Stop Data",
          "attribution" : "Berkeley Police Department",
          "averageRating" : 0,
          "category" : "Public Safety",
          "createdAt" : 1444171604,
          "description" : "This data was extracted from the Department’s Public Safety Server and covers the data beginning January 26, 2015.  On January 26, 2015 the department began collecting data pursuant to General Order B-4 (issued December 31, 2014).  Under that order, officers were required to provide certain data after making all vehicle detentions (including bicycles) and pedestrian detentions (up to five persons).  This data set lists stops by police in the categories of traffic, suspicious vehicle, pedestrian and bicycle stops.  Incident number, date and time, location and disposition codes are also listed in this data.\r\n\r\nAddress data has been changed from a specific address, where applicable, and listed as the block where the incident occurred.  Disposition codes were entered by officers who made the stop.  These codes included the person(s) race, gender, age (range), reason for the stop, enforcement action taken, and whether or not a search was conducted.\r\n\r\nThe officers of the Berkeley Police Department are prohibited from biased based policing, which is defined as any police-initiated action that relies on the race, ethnicity, or national origin rather than the behavior of an individual or information that leads the police to a particular individual who has been identified as being engaged in criminal activity.",


The `stops.json` file is clearly not a CSV file. In this case, the file contains data in the JSON (JavaScript Object Notation) format, a commonly used data format where data is recorded in a dictionary format. Python's [`json` module](https://docs.python.org/3/library/json.html) makes reading in this file as a dictionary simple.


```python
import json

# Note that this could cause our computer to run out of memory if the file
# is large. In this case, we've verified that the file is small enough to
# read in beforehand.
with open('data/stops.json') as f:
    stops_dict = json.load(f)

stops_dict.keys()
```




    dict_keys(['meta', 'data'])



Note that `stops_dict` is a Python dictionary, so displaying it will display the entire dataset in the notebook. This could cause the browser to crash, so we only display the keys of the dictionary above. To peek at the data without potentially crashing the browser, we can print the dictionary to a string and only output some of the first characters of the string.


```python
from pprint import pformat

def print_dict(dictionary, num_chars=1000):
    print(pformat(dictionary)[:num_chars])

print_dict(stops_dict['meta'])
```

    {'view': {'attribution': 'Berkeley Police Department',
              'averageRating': 0,
              'category': 'Public Safety',
              'columns': [{'dataTypeName': 'meta_data',
                           'fieldName': ':sid',
                           'flags': ['hidden'],
                           'format': {},
                           'id': -1,
                           'name': 'sid',
                           'position': 0,
                           'renderTypeName': 'meta_data'},
                          {'dataTypeName': 'meta_data',
                           'fieldName': ':id',
                           'flags': ['hidden'],
                           'format': {},
                           'id': -1,
                           'name': 'id',
                           'position': 0,
                           'renderTypeName': 'meta_data'},
                          {'dataTypeName': 'meta_data',
                           'fieldName': ':position',
                           'flags': ['hidden'],
                           'format': {},
                  



```python
print_dict(stops_dict['data'], num_chars=300)
```

    [[1,
      '29A1B912-A0A9-4431-ADC9-FB375809C32E',
      1,
      1444146408,
      '932858',
      1444146408,
      '932858',
      None,
      '2015-00004825',
      '2015-01-26T00:10:00',
      'SAN PABLO AVE / MARIN AVE',
      'T',
      'M',
      None,
      None],
     [2,
      '1644D161-1113-4C4F-BB2E-BF780E7AE73E',
      2,
      1444146408,
      '932858',
      14


We can likely deduce that the `'meta'` key in the dictionary contains a description of the data and its columns and the `'data'` contains a list of data rows. We can use this information to initialize a DataFrame.


```python
# Load the data from JSON and assign column titles
stops = pd.DataFrame(
    stops_dict['data'],
    columns=[c['name'] for c in stops_dict['meta']['view']['columns']])

stops
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
      <th>sid</th>
      <th>id</th>
      <th>position</th>
      <th>created_at</th>
      <th>...</th>
      <th>Incident Type</th>
      <th>Dispositions</th>
      <th>Location - Latitude</th>
      <th>Location - Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>29A1B912-A0A9-4431-ADC9-FB375809C32E</td>
      <td>1</td>
      <td>1444146408</td>
      <td>...</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1644D161-1113-4C4F-BB2E-BF780E7AE73E</td>
      <td>2</td>
      <td>1444146408</td>
      <td>...</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5338ABAB-1C96-488D-B55F-6A47AC505872</td>
      <td>3</td>
      <td>1444146408</td>
      <td>...</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
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
      <th>29205</th>
      <td>31079</td>
      <td>C2B606ED-7872-4B0B-BC9B-4EF45149F34B</td>
      <td>31079</td>
      <td>1496269085</td>
      <td>...</td>
      <td>T</td>
      <td>BM2TWN;</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>29206</th>
      <td>31080</td>
      <td>8FADF18D-7FE9-441D-8709-7BFEABDACA7A</td>
      <td>31080</td>
      <td>1496269085</td>
      <td>...</td>
      <td>T</td>
      <td>HM4TCS;</td>
      <td>37.8698757000001</td>
      <td>-122.286550846</td>
    </tr>
    <tr>
      <th>29207</th>
      <td>31081</td>
      <td>F60BD2A4-8C47-4BE7-B1C6-4934BE9DF838</td>
      <td>31081</td>
      <td>1496269085</td>
      <td>...</td>
      <td>1194</td>
      <td>AR;</td>
      <td>37.867207539</td>
      <td>-122.256529377</td>
    </tr>
  </tbody>
</table>
<p>29208 rows × 15 columns</p>
</div>




```python
# Prints column names
stops.columns
```




    Index(['sid', 'id', 'position', 'created_at', 'created_meta', 'updated_at',
           'updated_meta', 'meta', 'Incident Number', 'Call Date/Time', 'Location',
           'Incident Type', 'Dispositions', 'Location - Latitude',
           'Location - Longitude'],
          dtype='object')



The website contains documentation about the following columns:

| Column | Description | Type |
| ------ | ----------- | ---- |
| Incident Number | Number of incident created by Computer Aided Dispatch (CAD) program | Plain Text |
| Call Date/Time  | Date and time of the incident/stop | Date & Time |
| Location  | General location of the incident/stop | Plain Text |
| Incident Type | This is the occurred incident type created in the CAD program. A code signifies a traffic stop (T), suspicious vehicle stop (1196), pedestrian stop (1194) and bicycle stop (1194B). | Plain Text |
| Dispositions  | Ordered in the following sequence: 1st Character = Race, as follows: A (Asian) B (Black) H (Hispanic) O (Other) W (White) 2nd Character = Gender, as follows: F (Female) M (Male) 3rd Character = Age Range, as follows: 1 (Less than 18) 2 (18-29) 3 (30-39), 4 (Greater than 40) 4th Character = Reason, as follows: I (Investigation) T (Traffic) R (Reasonable Suspicion) K (Probation/Parole) W (Wanted) 5th Character = Enforcement, as follows: A (Arrest) C (Citation) O (Other) W (Warning) 6th Character = Car Search, as follows: S (Search) N (No Search) Additional dispositions may also appear. They are: P - Primary case report M - MDT narrative only AR - Arrest report only (no case report submitted) IN - Incident report FC - Field Card CO - Collision investigation report MH - Emergency Psychiatric Evaluation TOW - Impounded vehicle 0 or 00000 – Officer made a stop of more than five persons | Plain Text |
| Location - Latitude | General latitude of the call. This data is only uploaded after January 2017 | Number |
| Location - Longitude  | General longitude of the call. This data is only uploaded after January 2017. | Number |

Notice that the website doesn't contain descriptions for the first 8 columns of the `stops` table. Since these columns appear to contain metadata that we're not interested in analyzing this time, we drop them from the table.


```python
columns_to_drop = ['sid', 'id', 'position', 'created_at', 'created_meta',
                   'updated_at', 'updated_meta', 'meta']

# This function takes in a DF and returns a DF so we can use it for .pipe
def drop_unneeded_cols(stops):
    return stops.drop(columns=columns_to_drop)

stops.pipe(drop_unneeded_cols)
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
      <td>2015-01-26T00:10:00</td>
      <td>SAN PABLO AVE / MARIN AVE</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-00004829</td>
      <td>2015-01-26T00:50:00</td>
      <td>SAN PABLO AVE / CHANNING WAY</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-00004831</td>
      <td>2015-01-26T01:03:00</td>
      <td>UNIVERSITY AVE / NINTH ST</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
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
      <th>29205</th>
      <td>2017-00024245</td>
      <td>2017-04-30T22:59:26</td>
      <td>UNIVERSITY AVE/6TH ST</td>
      <td>T</td>
      <td>BM2TWN;</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>29206</th>
      <td>2017-00024250</td>
      <td>2017-04-30T23:19:27</td>
      <td>UNIVERSITY AVE /  WEST ST</td>
      <td>T</td>
      <td>HM4TCS;</td>
      <td>37.8698757000001</td>
      <td>-122.286550846</td>
    </tr>
    <tr>
      <th>29207</th>
      <td>2017-00024254</td>
      <td>2017-04-30T23:38:34</td>
      <td>CHANNING WAY /  BOWDITCH ST</td>
      <td>1194</td>
      <td>AR;</td>
      <td>37.867207539</td>
      <td>-122.256529377</td>
    </tr>
  </tbody>
</table>
<p>29208 rows × 7 columns</p>
</div>



As with the Calls dataset, we will answer the following three questions about the Stops dataset:

1. Are there missing values in the dataset?
1. Are there any missing values that were filled in (e.g. a 999 for unknown age or 12:00am for unknown date)?
1. Which parts of the data were entered by a human?

### Are there missing values?

We can clearly see that there are many missing latitude and longitudes. The data description states that these two columns are only filled in after Jan 2017.


```python
# True if row contains at least one null value
null_rows = stops.isnull().any(axis=1)

stops[null_rows]
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
      <td>2015-01-26T00:10:00</td>
      <td>SAN PABLO AVE / MARIN AVE</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-00004829</td>
      <td>2015-01-26T00:50:00</td>
      <td>SAN PABLO AVE / CHANNING WAY</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-00004831</td>
      <td>2015-01-26T01:03:00</td>
      <td>UNIVERSITY AVE / NINTH ST</td>
      <td>T</td>
      <td>M</td>
      <td>None</td>
      <td>None</td>
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
      <th>29078</th>
      <td>2017-00023764</td>
      <td>2017-04-29T01:59:36</td>
      <td>2180 M L KING JR WAY</td>
      <td>1194</td>
      <td>BM4IWN;</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>29180</th>
      <td>2017-00024132</td>
      <td>2017-04-30T12:54:23</td>
      <td>6TH/UNI</td>
      <td>1194</td>
      <td>M;</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>29205</th>
      <td>2017-00024245</td>
      <td>2017-04-30T22:59:26</td>
      <td>UNIVERSITY AVE/6TH ST</td>
      <td>T</td>
      <td>BM2TWN;</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>25067 rows × 7 columns</p>
</div>



We can check the other columns for missing values:


```python
# True if row contains at least one null value without checking
# the latitude and longitude columns
null_rows = stops.iloc[:, :-2].isnull().any(axis=1)

df_interact(stops[null_rows])
```


    A Jupyter Widget


    (63 rows, 7 columns) total


By browsing through the table above, we can see that all other missing values are in the Dispositions column. Unfortunately, we do not know from the data description why these Dispositions might be missing. Since only there are only 63 missing values compared to 25,000 rows in the original table, we can proceed with analysis while being mindful that these missing values could impact results.

### Are there any missing values that were filled in?

It doesn't seem like any previously missing values were filled in for us. Unlike in the Calls dataset where the date and time were in separate columns, the Call Date/Time column in the Stops dataset contains both date and time.

### Which parts of the data were entered by a human?

As with the Calls dataset, it looks like most of the columns in this dataset were recorded by a machine or were a category selected by a human (e.g. Incident Type).

However, the Location column doesn't have consistently entered values. Sure enough, we spot some typos in the data:


```python
stops['Location'].value_counts()
```




    2200 BLOCK SHATTUCK AVE            229
    37.8693028530001~-122.272234021    213
    UNIVERSITY AVE / SAN PABLO AVE     202
                                      ... 
    VALLEY ST / DWIGHT WAY               1
    COLLEGE AVE / SIXTY-THIRD ST         1
    GRIZZLY PEAK BLVD / MARIN AVE        1
    Name: Location, Length: 6393, dtype: int64



What a mess! It looks like sometimes an address was entered, sometimes a cross-street, and other times a latitude-longitude pair. Unfortunately, we don't have very complete latitude-longitude data to use in place of this column. We may have to manually clean this column if we want to use locations for future analysis.

We can also check the Dispositions column:


```python
dispositions = stops['Dispositions'].value_counts()

# Outputs a slider to pan through the unique Dispositions in
# order of how often they appear
interact(lambda row=0: dispositions.iloc[row:row+7],
         row=(0, len(dispositions), 7))
```


    A Jupyter Widget





    <function __main__.<lambda>>



The Dispositions columns also contains inconsistencies. For example, some dispositions start with a space, some end with a semicolon, and some contain multiple entries. The variety of values suggests that this field contains human-entered values and should be treated with caution.


```python
# Strange values...
dispositions.iloc[[0, 20, 30, 266, 1027]]
```




    M           1683
    M;           238
     M           176
    HF4TWN;       14
     OM4KWS        1
    Name: Dispositions, dtype: int64



In addition, the most common disposition is `M` which isn't a permitted first character in the Dispositions column. This could mean that the format of the column changed over time or that officers are allowed to enter in the disposition without matching the format in the data description. In any case, the column will be challenging to work with.

We can take some simple steps to clean the Dispositions column by removing leading and trailing whitespace, removing trailing semi-colons, and replacing the remaining semi-colons with commas.


```python
def clean_dispositions(stops):
    cleaned = (stops['Dispositions']
               .str.strip()
               .str.rstrip(';')
               .str.replace(';', ','))
    return stops.assign(Dispositions=cleaned)
```

As before, we can now pipe the `stops` DF through the cleaning functions we've defined:


```python
stops_final = (stops
               .pipe(drop_unneeded_cols)
               .pipe(clean_dispositions))
df_interact(stops_final)
```


    A Jupyter Widget


    (29208 rows, 7 columns) total


## Conclusion

As these two datasets have shown, data cleaning can often be both difficult and tedious. Cleaning 100% of the data often takes too long, but not cleaning the data at all results in faulty conclusions; we have to weigh our options and strike a balance each time we encounter a new dataset.

The decisions made during data cleaning impact all future analyses. For example, we chose not to clean the Location column of the Stops dataset so we should treat that column with caution. Each decision made during data cleaning should be carefully documented for future reference, preferably in a notebook so that both code and explanations appear together.


```python
# HIDDEN
# Save data to CSV for other chapters
# stops_final.to_csv('../ch5/data/stops.csv', index=False)
```
