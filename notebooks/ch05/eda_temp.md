

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

## Temporality

Temporality refers to how the data are situated in time and specifically to the date and time fields in the dataset. We seek to understand the following traits about these fields:

**What is the meaning of the date and time fields in the dataset?**

In the Calls and Stops dataset, the datetime fields represent when the call or stop was made by the police. However, the Stops dataset also originally had a datetime field recording when the case was entered into the database which we took out during data cleaning since we didn't think it would be useful for analysis.

In addition, we should be careful to note the timezone and Daylight Savings for datetime fields especially when dealing with data that comes from multiple locations.

**What representation do the date and time fields have in the data?**

Although the US uses the MM/DD/YYYY format, many other countries use the DD/MM/YYYY format. There are still more formats in use around the world and it's important to recognize these differences when analyzing data.

In the Calls and Stops dataset, the dates came in the MM/DD/YYYY format.

**Are there strange timestamps that might represent null values?**

Some programs use placeholder datetimes instead of null values. For example, Excel's default date is Jan 1st, 1990 and on Excel for Mac, it's Jan 1st, 1904. Many applications will generate a default datetime of 12:00am Jan 1st, 1970 or 11:59pm Dec 31st, 1969 since this is the [Unix Epoch for timestamps](https://www.wikiwand.com/en/Unix_time#/Encoding_time_as_a_number). If you notice multiple instances of these timestamps in your data, you should take caution and double check your data sources. Neither Calls nor Stops dataset contain any of these suspicious values.
