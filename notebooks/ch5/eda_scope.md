

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
calls = pd.read_csv('data/calls.csv', parse_dates=['EVENTDTTM'], infer_datetime_format=True)
stops = pd.read_csv('data/stops.csv', parse_dates=[1], infer_datetime_format=True)
```

## Scope

The scope of the dataset refers to the coverage of the dataset in relation to what we are interested in analyzing. We seek to answer the following question about our data scope:

**Does the data cover the topic of interest?**

For example, the Calls and Stops datasets contain call and stop incidents made in Berkeley. If we are interested in crime incidents in the state of California, however, these datasets will be too limited in scope.

In general, larger scope is more useful than smaller scope since we can filter larger scope down to a smaller scope but often can't go from smaller scope to larger scope. For example, if we had a dataset of police stops in the United States we could subset the dataset to investigate Berkeley.

Keep in mind that scope is a broad term not always used to describe geographic location. For example, it can also refer to time coverage — the Calls dataset only contains data for a 180 day period.

We will often address the scope of the dataset during the investigation of the data generation process and confirm the dataset's scope during EDA. Let's confirm the geographic and time scope of the Calls dataset.


```python
calls
```


```python
# Shows earliest and latest dates in calls
calls['EVENTDTTM'].dt.date.sort_values()
```


```python
calls['EVENTDTTM'].dt.date.max() - calls['EVENTDTTM'].dt.date.min()
```

The table contains data for a time period of 179 days which is close enough to the 180 day time period in the data description that we can suppose there were no calls on either April 14st, 2017 or August 29, 2017.

To check the geographic scope, we can use a map:


```python
import folium # Use the Folium Javascript Map Library
import folium.plugins

SF_COORDINATES = (37.87, -122.28)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = calls[['Latitude', 'Longitude']].astype('float').dropna().as_matrix()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)
```

With a few exceptions, the Calls dataset covers the Berkeley area. We can see that most police calls happened in the Downtown Berkeley and south of UC Berkeley campus areas.

Let's now confirm the temporal and geographic scope for the Stops dataset:


```python
stops
```


```python
stops['Call Date/Time'].dt.date.sort_values()
```

As promised, the data collection begins on January 26th, 2015. It looks like the data were downloaded somewhere around the beginning of May 2017 since the dates stop on April 30th, 2017. Let's draw a map to see the geographic data:


```python
SF_COORDINATES = (37.87, -122.28)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = stops[['Location - Latitude', 'Location - Longitude']].astype('float').dropna().as_matrix()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)
```

We can confirm that the police stops in the dataset happened in Berkeley, and that most police calls happened in the Downtown Berkeley and West Berkeley areas.
