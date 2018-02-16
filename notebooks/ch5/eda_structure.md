

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
from IPython.display import display, HTML

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 10
pd.options.display.max_columns = 8
```


```python
# HIDDEN
def display_two(df1, df2):
    '''Displays two DFs side-by-side.'''
    display(
        HTML('<div style="display: flex;">'
                 '{}'
                 '<div style="width: 20px;"></div>'
                 '{}'
             '</div>'.format(df1._repr_html_(), df2._repr_html_()))
    )
```

## Structure

The structure of a dataset refers to the "shape" of the data files. At a basic level, this refers to the format that the data are entered in. For example, we saw that the Calls dataset is a comma-separated values file:


```python
!head data/Berkeley_PD_-_Calls_for_Service.csv
```

The Stops dataset, on the other hand, is a JSON (JavaScript Object Notation) file.


```python
# Show first and last 5 lines of file
!head -n 5 data/stops.json
!echo '...'
!tail -n 5 data/stops.json
```

Of course, there are many other types of data formats. Here is a list of the most common formats:

- Comma-Separated Values (CSV) and Tab-Separated Values (TSV). These files contain tabular data delimited by either a comma for CSV or a tab character (`\t`) for TSV. These files are typically easy to work with because the data are entered in a similar format to DataFrames.
- JavaScript Object Notation (JSON). These files contain data in a nested dictionary format. Typically we have to read in the entire file as a Python dict and then figure out how to extract fields for a DataFrame from the dict.
- eXtensible Markup Language (XML) or HyperText Markup Language (HTML). These files also contain data in a nested format, for example:

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <note>
      <to>Tove</to>
      <from>Jani</from>
      <heading>Reminder</heading>
      <body>Don't forget me this weekend!</body>
    </note>
    ```
    
    In a later chapter we will use XPath to extract data from these types of files.
- Log data. Many applications will output some data as they run in an unstructured text format, for example:

    ```
    2005-03-23 23:47:11,663 - sa - INFO - creating an instance of aux_module.Aux
    2005-03-23 23:47:11,665 - sa.aux.Aux - INFO - creating an instance of Aux
    2005-03-23 23:47:11,665 - sa - INFO - created an instance of aux_module.Aux
    2005-03-23 23:47:11,668 - sa - INFO - calling aux_module.Aux.do_something
    2005-03-23 23:47:11,668 - sa.aux.Aux - INFO - doing something
    ```
    
    In a later chapter we will use Regular Expressions to extract data from these types of files.

## Joins

Data will often be split across multiple tables. For example, one table can describe some people's personal information while another will contain their emails:


```python
people = pd.DataFrame(
    [["Joey",      "blue",    42,  "M"],
     ["Weiwei",    "blue",    50,  "F"],
     ["Joey",      "green",    8,  "M"],
     ["Karina",    "green",    7,  "F"],
     ["Nhi",       "blue",     3,  "F"],
     ["Sam",       "pink",   -42,  "M"]], 
    columns = ["Name", "Color", "Number", "Sex"])

people
```


```python
email = pd.DataFrame(
    [["Deb",  "deborah_nolan@berkeley.edu"],
     ["Sam",  "samlau95@berkeley.edu"],
     ["John", "doe@nope.com"],
     ["Joey", "jegonzal@cs.berkeley.edu"],
     ["Weiwei", "weiwzhang@berkeley.edu"],
     ["Weiwei", "weiwzhang+123@berkeley.edu"],
     ["Karina", "kgoot@berkeley.edu"]], 
    columns = ["User Name", "Email"])

email
```

To match up each person with his or her email, we can join the two tables on the columns that contain the usernames. We must then decide what to do about people that appear in one table but not the other. For example, Fernando appears in the `people` table but not the `email` table. We have several types of joins for each strategy of matching missing values. One of the more common joins is the *inner join*, where any row that doesn't have a match is dropped in the final result:


```python
# Fernando, Nhi, Deb, and John don't appear
people.merge(email, how='inner', left_on='Name', right_on='User Name')
```

There are four basic joins that we use most often: inner, full (sometimes called "outer"), left, and right joins. Below is a diagram to show the difference between these types of joins.

![joins](https://github.com/DS-100/textbook/raw/master/assets/joins.png)

Use the dropdown menu below to show the result of the four different types of joins on the `people` and `email` tables. Notice which rows contain NaN values for outer, left, and right joins.


```python
# HIDDEN
def join_demo(join_type):
    display(HTML('people and email tables:'))
    display_two(people, email)
    display(HTML('<br>'))
    display(HTML('Joined table:'))
    display(people.merge(email, how=join_type,
                         left_on='Name', right_on='User Name'))
    
interact(join_demo, join_type=['inner', 'outer', 'left', 'right']);
```

## Structure Checklist

You should have answers to the following questions after looking at the structure of your datasets. We will answer them for the Calls and Stops datasets.

**Are the data in a standard format or encoding?**

Standard formats include:

- Tabular data: CSV, TSV, Excel, SQL
- Nested data: JSON, XML

The Calls dataset came in the CSV format while the Stops dataset came in the JSON format.

**Are the data organized in records (e.g. rows)? If not, can we define records by parsing the data?**

The Calls dataset came in rows; we extracted records from the Stops dataset.

**Are the data nested? If so, can we reasonably unnest the data?**

The Calls dataset wasn't nested; we didn't have to work too hard to unnest data from the Stops dataset.

**Do the data reference other data? If so, can we join the data?**

The Calls dataset references the day of week table. Joining those two tables gives us the day of week for each incident in the dataset. The Stops dataset had no obvious references.

**What are the fields (e.g. columns) in each record? What is the type of each column?**

The fields for the Calls and Stops datasets are described in the Data Cleaning sections for each dataset.
