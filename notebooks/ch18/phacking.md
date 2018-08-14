
**FROM DEB:**
- Testing many hypotheses on the same data set. (Maybe this could be a genetics example).

- Collecting many data sets and carrying out one test on each (A/B testing with ads)

- Entirely different people/institutions investigating a similar phenomenon - this is the example of Paul the Octopus.

- Making many decisions in the data collection, EDA process, and analysis process. This is "tip of the iceberg" phenomen.

**START SECTION**

⬇️⬇️⬇️⬇️


```python
### HIDDEN
# does this make it hidden?
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import pandas as pd
np.random.seed(0)
```

# P-hacking
As we discussed, a p-value or probability value is the chance, based on the model in the null hypothesis, that the test statistic is equal to the value that was observed in the data or is even further in the direction of the alternative. If a p-value is small, that means the tail beyond the observed statistic is small and so the observed statistic is far away from what the null predicts. This implies that the data support the alternative hypothesis better than they support the null. By convention, when we see that the p-value is below 0.05, the result is called statistically significant, and we reject the null hypothesis.

There are dangers that present itself when the p-value is misused. *P-hacking* is the act of misusing data analysis to show that patterns in data are statistically significant, when in reality they are not. This is often done by performing multiple tests on data and only focusing on the tests that return results that are significant. 

In this section, we will go over a few examples of the dangers of p-values and p-hacking.

## Multiple Hypothesis Testing


One of the biggest dangers of blindly relying on the p-value to determine "statistical significance" comes when we are just trying to find the "sexiest" results that give us "good" p-values. This is commonly done when doing "food frequency questionairres," or FFQs, to study eating habits' correlation with other characteristics (diseases, weight, etc). 
FiveThirtyEight, an online blog that focuses on opinion poll analysis among other things, made their own FFQ, and we can use their data to run our own analysis to find some silly results that can be considered "statistically significant."


```python
data = pd.read_csv('raw_anonymized_data.csv')

# These are the columns that give us characteristics of FFQ-takers
characteristics = data.columns[1:27]

# These are the columns that give us the quantities/frequencies of different food the FFQ-takers ate
ffq = data.columns[27:]
```


```python
data[characteristics].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cancer</th>
      <th>diabetes</th>
      <th>heart_disease</th>
      <th>belly</th>
      <th>ever_smoked</th>
      <th>currently_smoke</th>
      <th>smoke_often</th>
      <th>smoke_rarely</th>
      <th>never_smoked</th>
      <th>quit_smoking</th>
      <th>...</th>
      <th>neutralCable</th>
      <th>noCrash</th>
      <th>yesCrash</th>
      <th>uhCrash</th>
      <th>rash</th>
      <th>cat</th>
      <th>dog</th>
      <th>Dems</th>
      <th>atheist</th>
      <th>Jewish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Innie</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Outie</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Innie</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Innie</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Innie</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
data[ffq].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BREAKFASTSANDWICHFREQ</th>
      <th>BREAKFASTSANDWICHQUAN</th>
      <th>EGGSFREQ</th>
      <th>EGGSQUAN</th>
      <th>YOGURTFREQ</th>
      <th>YOGURTQUAN</th>
      <th>COTTAGECHEESEFREQ</th>
      <th>COTTAGECHEESEQUAN</th>
      <th>CREAMCHEESEFREQ</th>
      <th>CREAMCHEESEQUAN</th>
      <th>...</th>
      <th>DT_FIBER_INSOL</th>
      <th>DT_FIBER_SOL</th>
      <th>DT_PROT_ANIMAL</th>
      <th>DT_PROT_VEGETABLE</th>
      <th>DT_NITROGEN</th>
      <th>PHYTIC_ACID</th>
      <th>OXALIC_ACID</th>
      <th>COUMESTROL</th>
      <th>BIOCHANIN_A</th>
      <th>FORMONONETIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>...</td>
      <td>7.38</td>
      <td>1.25</td>
      <td>75.46</td>
      <td>16.00</td>
      <td>14.89</td>
      <td>365.70</td>
      <td>318.11</td>
      <td>0.0117</td>
      <td>0.0658</td>
      <td>0.00324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>9.11</td>
      <td>3.37</td>
      <td>59.41</td>
      <td>18.25</td>
      <td>12.51</td>
      <td>434.98</td>
      <td>112.66</td>
      <td>0.0107</td>
      <td>0.1390</td>
      <td>0.00743</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>...</td>
      <td>11.56</td>
      <td>4.74</td>
      <td>61.49</td>
      <td>28.46</td>
      <td>14.45</td>
      <td>606.43</td>
      <td>213.41</td>
      <td>0.0965</td>
      <td>0.0519</td>
      <td>0.00946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>26.34</td>
      <td>10.85</td>
      <td>28.71</td>
      <td>44.59</td>
      <td>12.15</td>
      <td>1570.07</td>
      <td>334.08</td>
      <td>0.2830</td>
      <td>0.0890</td>
      <td>0.01260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>16.48</td>
      <td>4.80</td>
      <td>32.41</td>
      <td>28.23</td>
      <td>9.80</td>
      <td>616.99</td>
      <td>422.55</td>
      <td>0.1630</td>
      <td>0.0994</td>
      <td>0.02070</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1066 columns</p>
</div>




```python
# Do some EDA on the data so that categorical values get changed to 1s and 0s
data.replace('Yes', 1, inplace=True)
data.replace('Innie', 1, inplace=True)
data.replace('No', 0, inplace=True)
data.replace('Outie', 0, inplace=True)

# Calculate the p value between every characteristic and food frequency/quantity
pvalues = {}
for c in characteristics:
    for f in ffq:
        pvalues[stat.pearsonr(data[c].tolist(), data[f].tolist())[1]] = (c, f)
```

    /Users/andrewkim/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:3021: RuntimeWarning: invalid value encountered in double_scalars
      r = r_num / r_den



```python
sorted_values = []
for key in sorted(pvalues.keys()):
    sorted_values += [(key, pvalues[key])]
```


```python
sorted_values[0:30]
```




    [(3.7392930595140498e-09, ('Jewish', 'GROUP_ALL_BRAN_ORIGINAL_TOTAL_GRAMS')),
     (3.7392930595140713e-09, ('Jewish', 'ALLBRANORIGTYPE')),
     (3.7392930595141143e-09,
      ('smoke_often', 'GROUP_COOK_FAT_ANIMAL_FAT_TOTAL_GRAMS')),
     (9.4530340028772084e-09,
      ('smoke_often', 'GROUP_LOW_CAL_KOOLAID_TOTAL_GRAMS')),
     (2.865639565547171e-08,
      ('Jewish', 'GROUP_ICE_CREAM_FROZEN_YOGURT_LOW_FAT_TOTAL_GRAMS')),
     (5.2203280690755088e-08,
      ('smoke_often', 'GROUP_STEAK_ROAST_FAT_ON_TOTAL_GRAMS')),
     (5.9917513301454779e-08, ('Jewish', 'GROUP_RAW_TOMATOES_TOTAL_GRAMS')),
     (6.0267821160795707e-07,
      ('smoke_often', 'GROUP_CORN_FLAKES_CORN_PUFFS_TOTAL_GRAMS')),
     (4.5182117663961682e-06, ('smoke_often', 'F18D1TN7')),
     (8.0499444720125634e-06, ('smoke_rarely', 'FLAXHEMPSEEDOILTYPE')),
     (8.0499444720126362e-06, ('smoke_rarely', 'GROUP_CHEX_OTHER_TOTAL_GRAMS')),
     (8.0499444720126701e-06, ('smoke_rarely', 'GROUP_LIFE_TOTAL_GRAMS')),
     (8.8030660205637771e-06, ('smoke_often', 'GROUP_PORK_FAT_ON_TOTAL_GRAMS')),
     (1.1446426335225304e-05, ('smoke_rarely', 'F18D3CN3')),
     (1.6158156614271367e-05, ('smoke_rarely', 'GROUP_SOY_MILK_TOTAL_GRAMS')),
     (2.1945383378396457e-05, ('smoke_rarely', 'SUP_OM_6')),
     (2.8476184473490123e-05, ('dog', 'EGGROLLQUAN')),
     (3.1447466606889132e-05, ('smoke_rarely', 'F20D3N3')),
     (3.9156738423019172e-05, ('smoke_often', 'GROUP_BAGELS_WHITE_TOTAL_GRAMS')),
     (5.4472799562211073e-05,
      ('smoke_often', 'GROUP_HOT_DOG_BEEF_OR_PORK_TOTAL_GRAMS')),
     (5.4976262450193749e-05, ('Jewish', 'V_REDOR_TOMATO')),
     (6.6642014484263388e-05, ('currently_smoke', 'ENERGYDRINKSQUAN')),
     (7.2255229976485587e-05, ('smoke_rarely', 'CAMD5')),
     (7.7299724934785873e-05, ('smoke_often', 'LATINO')),
     (8.1320697785677561e-05, ('smoke_often', 'BUTTERQUAN')),
     (0.00010682433178383818, ('currently_smoke', 'BEERQUAN')),
     (0.00012496025027669932,
      ('quit_smoking', 'GROUP_WHOLE_GRAIN_CRACKERS_LOW_FAT_TOTAL_GRAMS')),
     (0.0001386176772447345, ('mathReading', 'OTHERCHIPSFREQ')),
     (0.00014468096634093775, ('smoke_often', 'F18D2CLA')),
     (0.00017717189454132738, ('rash', 'SODAQUAN'))]



## A/B Testing
A/B testing is a very simple concept. We measure a statistic in a normal, controlled environment (we'll call this A), and then we compare that to the same statistic in an environment with *one* change. This form of testing is used frequently in marketing and ad research to compare the effectiveness of certain features of ads. 

Let's say we are working for a company whose website lets users make their own custom videogames. The company has a free version, which lets users make very basic videogames, and a paid version, which gives users access to more advanced tools for making videogames. When a user has finished making a videogame via a free account, we send them to a landing page that gives them the option to sign up for a paid account. Our measured statistic in this case would be how many free users sign up for a paid account upon reaching this page. We can send half of our users one version of the page, which may have text explaining in detail the benefits of the paid account (this will be version A), and the other half of our users will get another version of the page, which may have a colorful graphic that explains some of the benefits of the paid account (this will be version B). 

There is a very specific reason why it's called A/B testing, and not A/B/C/D... testing. That is because we can very easily run into problems if we try to test multiple versions at the same time. 

Let's say that we have 15 different sign up pages (one is the control, in this case "A"), each with something different about them (one has a picture of a puppy, one has a quote from a customer, one has a graphic, etc.), and let's say that in this case, none of our variations actually has an effect on user interaction (so we can use a Gaussian distribution with a mean of 0 and a std of 0.1).


```python
n = 50
reps = 1000
num_pages = 15

landing_pages = [np.random.normal(0, 0.1, n) for _ in range(num_pages)]

# This will be our "control"
A = landing_pages[0]

# Our test statistic will be the difference between the mean percentage 
def test_stat(A, B):
    return np.abs(np.mean(B) - np.mean(A))

def permute(A, B):
    combined = np.append(A, B)
    shuffled = np.random.choice(combined, size=len(combined), replace=False)
    return shuffled[:n], shuffled[n:]

p_vals = []
for i in range(1, num_pages):
    B = landing_pages[i]
    obs = test_stat(A, B)
    resampled = [test_stat(*permute(A, B)) for _ in range(reps)]
    p_val = np.count_nonzero(obs >= resampled) / reps
    p_vals.append(p_val)
p_vals
```




    [0.434,
     0.208,
     0.01,
     0.02,
     0.631,
     0.171,
     0.222,
     0.212,
     0.276,
     0.772,
     0.006,
     0.001,
     0.572,
     0.593]



As we can see, more than one of these ads seems to have p-values less than 0.05, despite our knowing that there actually no difference between the pages. This is why we do single A/B testing with multiple trials, as opposed to multiple hypothesis testing with only single trials. It is too easy for a p-value to give us a false positive if we just try a bunch of times.

## Many Tests for One Phenomenon
Sometimes, multiple testing can happen by accident. If many researchers are investigating the same phenomenon at the same time, then it's very possible that one of the researchers can end up with a lucky trial. That is exactly what happened during the 2010 World Cup.

### Paul the Octopus

Paul the Octopus was a common octopus who lived in a Sea Life Centre in Oberhausen, Germany. He is most well known for correctly guessing all seven soccer matches Germany played during the 2010 World Cup, as well as the final match, which was between Netherlands and Spain. 

Before a match was played, Paul's owners would place two boxes in his tank containing food, each box labeled with a different flag of the opposing countries. Whichever box Paul chose to eat from first was considered his prediction for the outcome of the match. 

<img src="https://news.bbcimg.co.uk/media/images/49659000/jpg/_49659323_octopus.jpg" width="400" />

So why was Paul so good at predicting the outcome of these matches? Was he actually psychic, or was he just lucky? We might ask what’s the chance he got all of the predictions correct, assuming he was just “guessing”?

Paul correctly predicted 8 of the 2010 World Cup games, each time he had a 1/2 chance of making the correct prediction. The one way to get all 8 matches correct out of 8 is:
$$(1/2)^8 = 1/256$$

So was he actually psychic? Or is there something more to uncover?

Turns out, there were tons of animals (some of them in the same zoo as Paul!) doing the same thing, trying to guess the outcome of their respective home countries' matches, including:
- Mani the Parakeet, from Singapore
- Leon the Porcupine, from Germany
- Petty the Pygmy Hippopotamus, from Germany
- Otto Armstrong the Octopus, from Germany
- Anton the Tamarin, from Germany
- Jimmy the Peruvian Guinea Pig, from Germany
- Xiaoge the Octopus, from China
- Pauline the Octopus, from the Netherlands
- Pino the Chimpanzee, from Estonia
- Apelsin the Red River Hog, from Estonia
- Harry the Crocodile, from Australia
None of whom got them all right (although Mani the Parakeet got 7 matches out of 8 right).

Some might argue that getting them all wrong would also be remarkable. So what are the chances that at least one of the 12 animals would get either all right or all wrong? 

We can use simple probability to figure this out. We have 12 trials (in this case, animals), where each independent trial has a $2*(1/2)^8 = 1/128$ chance of getting all predictions right or wrong. So what is the probability of having *at least* one success? That's $1 - P_{all \textrm{ }failures} = 1 - (127/128)^{12} = 1 - 0.910 = 0.090$

We have an 9% chance of getting an animal that will select all of the right predictions, and that's not including all of the animals in the world that were also doing these "predictions." That's not that rare - it's the dangers of multiple testing that caused this "phenomenon." This one octopus out of many different animals in the world happened to have guessed all of the right predictions, and the popularity of the situation caused it to become magical.

To those of you wondering if it really was luck, it has been shown that the species *Octopus vulgaris* is actually colorblind, and some believe that octopuses are drawn to horizontal shapes, hence Paul's decision to choose Germany, except when playing against Spain and Serbia.

In the end, we know that studies are more trustworthy when they are replicated. Data scientists should try to avoid cases like Paul the Octopus's where there has only been one real case of the animal correctly predicting a bunch of World Cup matches in a row. Only when we see him doing that for multiple soccer tournaments should we start looking at the data.

## P-Hacking is just the tip of the iceberg

As it turns out, p-hacking isn't the only thing data scientists and statisticians have to worry about when making sound inferences from data. There are many stages to the design and analysis of a successful study, as shown below (from Leek & Peng's *P values are just the tip of the iceberg*).

<img src='https://www.nature.com/polopoly_fs/7.25671.1429983882!/image/P1.jpg_gen/derivatives/landscape_300/P1.jpg'>

As shown, the last step of the whole "data pipeline" is the calculation of an inferential statistic like the p-value, and having a rule applied to it (e.g. p > 0.05). But there are many other decisions that are made beforehand, like experimental design or EDA, that can have much greater effects on the results - mistakes like simple rounding or measurement errors, choosing the wrong model, or not taking into account confounding factors can change everything. By changing the way data are cleaned, summarized, or modeled, we can achieve arbitrary levels of statistical significance.

A simple example of this would be in the case of rolling a pair of dice and getting two 6s. If we were to take a null hypothesis that the dice are fair and not weighted, and take our test statistic to be the sum of the dice, we will find that the p-value of this outcome will be 1/36 or 0.028, and gives us statistically signficant results that the dice are fair. But obviously, a single roll is not nearly enough rolls to provide us with good evidence to say whether the results are statistically significant or not, and shows that blindly applying the p-value without properly designing a good experiment can result in bad results.

In the end, what is most important is education on the subject of safe hypothesis testing, and making sure you don't fall into the follies of poor statistical decisions.
