

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
```

## The Students of Data 100

Recall that the data science lifecycle involves the following broad steps:

1. **Question/Problem Formulation:** 
    1. What do we want to know or what problems are we trying to solve?  
    1. What are our hypotheses? 
    1. What are our metrics of success? <br/><br/>
1. **Data Acquisition and Cleaning:** 
    1. What data do we have and what data do we need?  
    1. How will we collect more data? 
    1. How do we organize the data for analysis?  <br/><br/>
1. **Exploratory Data Analysis:** 
    1. Do we already have relevant data?  
    1. What are the biases, anomalies, or other issues with the data?  
    1. How do we transform the data to enable effective analysis? <br/><br/>
1. **Prediction and Inference:** 
    1. What does the data say about the world?  
    1. Does it answer our questions or accurately solve the problem?  
    1. How robust are our conclusions? <br/><br/>
    

## Question Formulation

We would like to figure out if the data we have on student names in Data 100 give
us any additional information about the students themselves. Although this is a
vague question to ask, it is enough to get us working with our data and we can
surely make the question more precise as we go.

## Data Acquisition and Cleaning

**In Data 100, we will study various methods to collect data.**

Let's begin by looking at our data, the roster of student first names that we've downloaded from a previous offering of Data 100.

Don't worry if you don't understand the code for now; we'll introduce the libraries in more depth later. Instead, focus on the process and the charts that we show.


```python
import pandas as pd

students = pd.read_csv('roster.csv')
students
```

We can quickly see that there are some quirks in the data. For example, one of the student's names is all uppercase letters. In addition, it is not obvious what the Role column is for.

**In Data 100, we will study how to identify anomalies in data and apply corrections.** The differences in capitalization will cause our programs to think that `'BRYAN'` and `'Bryan'` are different names when they are identical for our purposes. Let's convert all names to lower case to avoid this.


```python
students['Name'] = students['Name'].str.lower()
students
```

Now that our data are in a format that's easier for us to work with, let's proceed to exploratory data analysis.
