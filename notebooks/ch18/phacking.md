
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
        pvalues[stat.ttest_ind(data[c], data[f])[1]] = (c, f)
```


```python
sorted_values = []
for key in sorted(pvalues.keys()):
    sorted_values += [(key, pvalues[key])]
sorted_values[100:150]
```




    [(6.358126097548407e-63, ('readingMath', 'VITAMINEQUAN')),
     (6.6214062008543634e-63, ('rash', 'OTHERSOUPQUAN')),
     (7.6903104344215803e-63, ('Jewish', 'GROUP_SOLID_COUNT')),
     (7.924293507849059e-63, ('smoke_rarely', 'GROUP_SOLID_COUNT')),
     (8.4095237986025954e-63, ('currently_smoke', 'GROUP_SOLID_COUNT')),
     (8.9183757464713533e-63, ('yesCrash', 'GROUP_SOLID_COUNT')),
     (8.9662001457940954e-63, ('Jewish', 'LIVERQUAN')),
     (9.0613706048428091e-63, ('unfavCable', 'VITAMINEQUAN')),
     (9.7275141675628686e-63, ('rash', 'GROUP_SOLID_COUNT')),
     (1.0009811370750712e-62, ('favCable', 'GROUP_SOLID_COUNT')),
     (1.1204243469978213e-62, ('diabetes', 'GROUP_SOLID_COUNT')),
     (1.1519587976886106e-62, ('readingMath', 'GROUP_SOLID_COUNT')),
     (1.1841793766753675e-62, ('unfavCable', 'GROUP_SOLID_COUNT')),
     (1.2170942152162999e-62, ('noCrash', 'GROUP_SOLID_COUNT')),
     (1.2507113140199772e-62, ('dog', 'GROUP_SOLID_COUNT')),
     (1.2749047717364511e-62, ('noCrash', 'VITAMINEQUAN')),
     (1.2850385340105879e-62, ('cat', 'GROUP_SOLID_COUNT')),
     (1.3923572355954996e-62, ('mathReading', 'GROUP_SOLID_COUNT')),
     (1.4429807111480728e-62, ('Jewish', 'MINERALSYESORNO')),
     (1.4675906055414749e-62, ('neutralCable', 'GROUP_SOLID_COUNT')),
     (1.4921444186594713e-62, ('Jewish', 'PUDDINGQUAN')),
     (1.5463783306000627e-62, ('favCable', 'LENTILSOUPQUAN')),
     (1.5861102830912664e-62, ('cancer', 'GROUP_SOLID_COUNT')),
     (1.6271545209384683e-62, ('uhCrash', 'GROUP_SOLID_COUNT')),
     (1.7710512714167032e-62, ('dog', 'VITAMINEQUAN')),
     (1.9365728037394257e-62, ('atheist', 'GROUP_SOLID_COUNT')),
     (2.0322275434385771e-62, ('never_smoked', 'GROUP_SOLID_COUNT')),
     (2.1818641127738961e-62, ('Dems', 'GROUP_SOLID_COUNT')),
     (2.4212479707245423e-62, ('currently_smoke', 'MEATSUBSTITUTESQUAN')),
     (2.4294027322207792e-62, ('cat', 'VITAMINEQUAN')),
     (2.6173843137025856e-62, ('belly', 'GROUP_SOLID_COUNT')),
     (2.675567336257232e-62, ('right_hand', 'GROUP_SOLID_COUNT')),
     (2.9200619403975e-62, ('smoke_rarely', 'BANANASQUAN')),
     (3.6846874183612466e-62, ('Jewish', 'CANNEDFRUITQUAN')),
     (3.9833035518075475e-62, ('Jewish', 'SPAGHETTIQUAN')),
     (4.9112811758307594e-62, ('favCable', 'OTHERSOUPQUAN')),
     (5.8180405698122421e-62, ('mathReading', 'VITAMINEQUAN')),
     (9.7908755356540434e-62, ('neutralCable', 'VITAMINEQUAN')),
     (1.9501696385567028e-61, ('cancer', 'VITAMINEQUAN')),
     (2.031802496631482e-61, ('smoke_often', 'COLDCEREALQUAN')),
     (2.0318024966315393e-61, ('Jewish', 'COLDCEREALQUAN')),
     (2.126328127399533e-61, ('rash', 'VEGETABLESOUPQUAN')),
     (2.3946563950587075e-61, ('uhCrash', 'VITAMINEQUAN')),
     (2.670114135628953e-61, ('yesCrash', 'PEACHESQUAN')),
     (3.7718701944868199e-61, ('smoke_rarely', 'SPAGHETTIQUAN')),
     (7.1582767067797031e-61, ('atheist', 'VITAMINEQUAN')),
     (8.7561199803539314e-61, ('never_smoked', 'VITAMINEQUAN')),
     (9.7235378859665905e-61, ('right_hand', 'VITAMINEQUAN')),
     (1.0404958035361395e-60, ('belly', 'VITAMINEQUAN')),
     (1.0769332382521734e-60, ('Dems', 'VITAMINEQUAN'))]



## A/B Testing
A/B testing is a very simple concept. We measure a statistic in a normal, controlled environment (we'll call this A), and then we compare that to the same statistic in an environment with *one* change. This form of testing is used frequently in marketing and ad research to compare the effectiveness of certain features of ads. 

Let's say we are working for a company whose website lets users make their own custom videogames. The company has a free version, which lets users make very basic videogames, and a paid version, which gives users access to more advanced tools for making videogames. When a user has finished making a videogame via a free account, we send them to a landing page that gives them the option to sign up for a paid account. Our measured statistic in this case would be how many free users sign up for a paid account upon reaching this page. We can send half of our users one version of the page, which may have text explaining in detail the benefits of the paid account (this will be version A), and the other half of our users will get another version of the page, which may have a colorful graphic that explains some of the benefits of the paid account (this will be version B). 

There is a very specific reason why it's called A/B testing, and not A/B/C/D... testing. That is because we can very easily run into problems if we try to test multiple versions at the same time. 

Let's say that we have 15 different sign up pages (one is the control, in this case "A"), each with something different about them (one has a picture of a puppy, one has a quote from a customer, one has a graphic, etc.), and let's say that in this case, none of our variations actually has an effect on user interaction (so we can use a Gaussian distribution with a mean of 0 and a std of 0.1).


```python
landing_pages = [np.random.normal(0, 0.1, 50) for _ in range(15)] # can't seem to get the seed to work??
A = landing_pages[0]
pvalues = []
for i in range(1, 15):
    B = landing_pages[i]
    observed_differences = A - B
    differences = A - np.random.normal(0, 0.1, 50)
    pvalues = [sum(observed_differences >= differences) / 50]
min(pvalues)
```




    0.44



As we can see, more than one of these ads seems to have exceeded a 5% difference in user engagement, despite our knowing that there actually no difference. This is why we do single A/B testing with multiple trials, as opposed to multiple hypothesis testing with only single trials.

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
