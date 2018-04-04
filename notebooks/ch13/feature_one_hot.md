
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Regression-on-Categorical-Data" data-toc-modified-id="Regression-on-Categorical-Data-1">Regression on Categorical Data</a></span></li></ul></div>


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

## Regression on Categorical Data

In 2014, Walmart released some of its sales data as part of a competition to predict the weekly sales of its stores. We've taken a subset of their data and loaded it below.


```python
walmart = pd.read_csv('walmart.csv')
walmart
```

The data contains several interesting features, including whether a week contained a holiday (`IsHoliday`), the unemployment rate that week (`Unemployment`), and whether the store had any special deals that week (`MarkDown`).

Our goal is to create a model that predicts the `Weekly_Sales` variable using the other variables in our data. Using a linear regression model, we can use the `Temperature`, `Fuel_Price`, and `Unemployment` columns because contain numerical data. However, the `IsHoliday` and `MarkDown` seem useful. For example, the median weekly sales is a bit higher during holidays.


```python
sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=walmart);
```

The different markdown categories seem to correlate with different weekly sale amounts well.


```python
markdowns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
plt.figure(figsize=(7, 5))
sns.pointplot(x='MarkDown', y='Weekly_Sales', data=walmart, order=markdowns);
```

In order to use 
