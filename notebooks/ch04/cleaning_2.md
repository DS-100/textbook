
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Cleaning-The-Stops-Dataset" data-toc-modified-id="Cleaning-The-Stops-Dataset-1">Cleaning The Stops Dataset</a></span><ul class="toc-item"><li><span><a href="#Are-there-missing-values?" data-toc-modified-id="Are-there-missing-values?-1.1">Are there missing values?</a></span></li><li><span><a href="#Are-there-any-missing-values-that-were-filled-in?" data-toc-modified-id="Are-there-any-missing-values-that-were-filled-in?-1.2">Are there any missing values that were filled in?</a></span></li><li><span><a href="#Which-parts-of-the-data-were-entered-by-a-human?" data-toc-modified-id="Which-parts-of-the-data-were-entered-by-a-human?-1.3">Which parts of the data were entered by a human?</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-2">Conclusion</a></span></li></ul></div>


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
pd.options.display.max_columns = 8
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

Note that `stops_dict` is a Python dictionary, so displaying it will display the entire dataset in the notebook. This could cause the browser to crash, so we only display the keys of the dictionary above. To peek at the data without potentially crashing the browser, we can print the dictionary to a string and only output some of the first characters of the string.


```python
from pprint import pformat

def print_dict(dictionary, num_chars=1000):
    print(pformat(dictionary)[:num_chars])

print_dict(stops_dict['meta'])
```


```python
print_dict(stops_dict['data'], num_chars=300)
```

We can likely deduce that the `'meta'` key in the dictionary contains a description of the data and its columns and the `'data'` contains a list of data rows. We can use this information to initialize a DataFrame.


```python
# Load the data from JSON and assign column titles
stops = pd.DataFrame(
    stops_dict['data'],
    columns=[c['name'] for c in stops_dict['meta']['view']['columns']])

stops
```


```python
# Prints column names
stops.columns
```

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

We can check the other columns for missing values:


```python
# True if row contains at least one null value without checking
# the latitude and longitude columns
null_rows = stops.iloc[:, :-2].isnull().any(axis=1)

df_interact(stops[null_rows])
```

By browsing through the table above, we can see that all other missing values are in the Dispositions column. Unfortunately, we do not know from the data description why these Dispositions might be missing. Since only there are only 63 missing values compared to 25,000 rows in the original table, we can proceed with analysis while being mindful that these missing values could impact results.

### Are there any missing values that were filled in?

It doesn't seem like any previously missing values were filled in for us. Unlike in the Calls dataset where the date and time were in separate columns, the Call Date/Time column in the Stops dataset contains both date and time.

### Which parts of the data were entered by a human?

As with the Calls dataset, it looks like most of the columns in this dataset were recorded by a machine or were a category selected by a human (e.g. Incident Type).

However, the Location column doesn't have consistently entered values. Sure enough, we spot some typos in the data:


```python
stops['Location'].value_counts()
```

What a mess! It looks like sometimes an address was entered, sometimes a cross-street, and other times a latitude-longitude pair. Unfortunately, we don't have very complete latitude-longitude data to use in place of this column. We may have to manually clean this column if we want to use locations for future analysis.

We can also check the Dispositions column:


```python
dispositions = stops['Dispositions'].value_counts()

# Outputs a slider to pan through the unique Dispositions in
# order of how often they appear
interact(lambda row=0: dispositions.iloc[row:row+7],
         row=(0, len(dispositions), 7))
```

The Dispositions columns also contains inconsistencies. For example, some dispositions start with a space, some end with a semicolon, and some contain multiple entries. The variety of values suggests that this field contains human-entered values and should be treated with caution.


```python
# Strange values...
dispositions.iloc[[0, 20, 30, 266, 1027]]
```

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

## Conclusion

As these two datasets have shown, data cleaning can often be both difficult and tedious. Cleaning 100% of the data often takes too long, but not cleaning the data at all results in faulty conclusions; we have to weigh our options and strike a balance each time we encounter a new dataset.

The decisions made during data cleaning impact all future analyses. For example, we chose not to clean the Location column of the Stops dataset so we should treat that column with caution. Each decision made during data cleaning should be carefully documented for future reference, preferably in a notebook so that both code and explanations appear together.


```python
# HIDDEN
# Save data to CSV for other chapters
stops_final.to_csv('../ch5/data/stops.csv', index=False)
```
