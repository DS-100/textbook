
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
        pvalues[stat.ttest_ind(data[c].tolist(), data[f].tolist(), equal_var=False)[1]] = (c, f)
```


```python
sorted_values = []
for key in sorted(pvalues.keys()):
    sorted_values += [(key, pvalues[key])]
```


```python
sorted_values[2000:2100]
```




    [(2.3648241474577024e-26, ('Jewish', 'FRIESFREQ')),
     (2.3793938164610517e-26, ('readingMath', 'EGGSQUAN')),
     (2.4257035809442306e-26, ('smoke_rarely', 'ASH')),
     (2.5172312265496039e-26, ('right_hand', 'GROUP_SUGARYBEVG_TOTAL_FREQUENCY')),
     (2.544424004492252e-26, ('favCable', 'FRIESFREQ')),
     (2.5763300883291532e-26, ('currently_smoke', 'ASH')),
     (2.6205744857515174e-26, ('belly', 'VEGGIESFREQ')),
     (2.6365909377739635e-26, ('cancer', 'COOKEDGREENSQUAN')),
     (2.6652602262111569e-26, ('mathReading', 'MAYOQUAN')),
     (2.7289075788443869e-26, ('uhCrash', 'BROCCOLIFREQ')),
     (2.7431115431268559e-26, ('yesCrash', 'ASH')),
     (2.8108805895937193e-26, ('cancer', 'BERRIESQUAN')),
     (2.845061863498948e-26, ('belly', 'V_REDOR_OTHER')),
     (2.9205917867356232e-26, ('belly', 'FATRNP')),
     (3.0137708614710861e-26, ('Jewish', 'SLOWWALKTIME')),
     (3.0137708614710866e-26, ('smoke_often', 'SLOWWALKTIME')),
     (3.0275989228139054e-26, ('rash', 'ASH')),
     (3.1014395638929791e-26, ('right_hand', 'PSVEGPOT')),
     (3.1326215198600197e-26, ('favCable', 'ASH')),
     (3.1366890856379454e-26, ('Jewish', 'SLICEDCHEESEFREQ')),
     (3.1669719082216128e-26, ('smoke_rarely', 'SLICEDCHEESEFREQ')),
     (3.2610107414345672e-26, ('favCable', 'DT_VITB6')),
     (3.3064228544084095e-26, ('currently_smoke', 'SLICEDCHEESEFREQ')),
     (3.4001591380551358e-26, ('currently_smoke', 'SLOWWALKTIME')),
     (3.4090170735731775e-26, ('noCrash', 'BARBECUESAUCEQUAN')),
     (3.558184613002788e-26, ('yesCrash', 'SLICEDCHEESEFREQ')),
     (3.6073566135001483e-26, ('Dems', 'OYSTERSQUAN')),
     (3.6118323233868756e-26, ('diabetes', 'ASH')),
     (3.675704444013712e-26, ('right_hand', 'VEGGIESFREQ')),
     (3.7481787661825261e-26, ('readingMath', 'ASH')),
     (3.8919317883079915e-26, ('unfavCable', 'ASH')),
     (4.0435325616942995e-26, ('noCrash', 'ASH')),
     (4.0837038500260336e-26, ('right_hand', 'DT_M201')),
     (4.0911872483880518e-26, ('uhCrash', 'TUNAQUAN')),
     (4.146771395569101e-26, ('right_hand', 'STEAKQUAN')),
     (4.1884274452317456e-26, ('rash', 'SLICEDCHEESEFREQ')),
     (4.2034519784050791e-26, ('dog', 'ASH')),
     (4.2809736847197643e-26, ('diabetes', 'FRIESFREQ')),
     (4.291581975156661e-26, ('yesCrash', 'HICQUAN')),
     (4.3721927964005876e-26, ('cat', 'ASH')),
     (4.3921253128321055e-26, ('unfavCable', 'CREAMCHEESEQUAN')),
     (4.4350825316464269e-26, ('diabetes', 'DT_THIA')),
     (4.4817104555037101e-26, ('favCable', 'SLICEDCHEESEFREQ')),
     (4.4826970694501157e-26, ('mathReading', 'LIGHTHOUSETIME')),
     (4.5467487274008567e-26, ('readingMath', 'OTHERBREADSQUAN')),
     (4.6180018643021147e-26, ('unfavCable', 'EGGSQUAN')),
     (4.8728166592136976e-26, ('uhCrash', 'COOKEDGREENSQUAN')),
     (4.9368990813839132e-26, ('mathReading', 'ASH')),
     (5.0514148130974568e-26, ('readingMath', 'FRIESFREQ')),
     (5.3683522429915653e-26, ('neutralCable', 'ASH')),
     (5.6441805479086806e-26, ('mathReading', 'CABBAGEQUAN')),
     (5.6976036778089817e-26, ('uhCrash', 'BERRIESQUAN')),
     (6.0359587858911605e-26, ('unfavCable', 'FRIESFREQ')),
     (6.1125482005747637e-26, ('cancer', 'ASH')),
     (6.1497319883394071e-26, ('atheist', 'CARROTSQUAN')),
     (6.1566700228749851e-26, ('Jewish', 'DT_MAGN')),
     (6.1712323081055326e-26, ('smoke_rarely', 'DT_MAGN')),
     (6.2004926814671537e-26, ('currently_smoke', 'DT_MAGN')),
     (6.2299351623111835e-26, ('yesCrash', 'DT_MAGN')),
     (6.2487573965510624e-26, ('diabetes', 'SLICEDCHEESEFREQ')),
     (6.2744430777919058e-26, ('rash', 'DT_MAGN')),
     (6.2893714664074544e-26, ('favCable', 'DT_MAGN')),
     (6.3438977483978232e-26, ('Jewish', 'DT_KCAL')),
     (6.3466106448157961e-26, ('smoke_rarely', 'DT_KCAL')),
     (6.3495513560086257e-26, ('diabetes', 'DT_MAGN')),
     (6.3520409699339736e-26, ('currently_smoke', 'DT_KCAL')),
     (6.3574773442942995e-26, ('yesCrash', 'DT_KCAL')),
     (6.3647137161203254e-26, ('readingMath', 'DT_MAGN')),
     (6.3656432644580298e-26, ('rash', 'DT_KCAL')),
     (6.3683682705416912e-26, ('favCable', 'DT_KCAL')),
     (6.3792834832606428e-26, ('diabetes', 'DT_KCAL')),
     (6.3799233550283297e-26, ('unfavCable', 'DT_MAGN')),
     (6.3820160882132378e-26, ('readingMath', 'DT_KCAL')),
     (6.384750215748186e-26, ('unfavCable', 'DT_KCAL')),
     (6.3874858668028671e-26, ('noCrash', 'DT_KCAL')),
     (6.3898668706358933e-26, ('uhCrash', 'ASH')),
     (6.3902230423156149e-26, ('dog', 'DT_KCAL')),
     (6.3926557979670842e-26, ('diabetes', 'ICECREAMTYPE')),
     (6.3929617432252591e-26, ('cat', 'DT_KCAL')),
     (6.3951804356665366e-26, ('noCrash', 'DT_MAGN')),
     (6.401187007734326e-26, ('mathReading', 'DT_KCAL')),
     (6.4066781616334502e-26, ('neutralCable', 'DT_KCAL')),
     (6.4104851215882502e-26, ('dog', 'DT_MAGN')),
     (6.4149263776693737e-26, ('cancer', 'DT_KCAL')),
     (6.4176788495066232e-26, ('uhCrash', 'DT_KCAL')),
     (6.42583757696944e-26, ('cat', 'DT_MAGN')),
     (6.4369891803709244e-26, ('atheist', 'DT_KCAL')),
     (6.4425202766307165e-26, ('never_smoked', 'DT_KCAL')),
     (6.4508284985823945e-26, ('Dems', 'DT_KCAL')),
     (6.4721832110152647e-26, ('mathReading', 'DT_MAGN')),
     (6.4730518605868334e-26, ('belly', 'DT_KCAL')),
     (6.4758367631659466e-26, ('right_hand', 'DT_KCAL')),
     (6.5033221858021692e-26, ('neutralCable', 'DT_MAGN')),
     (6.5503968290410524e-26, ('cancer', 'DT_MAGN')),
     (6.5661867014172405e-26, ('uhCrash', 'DT_MAGN')),
     (6.6385266259568142e-26, ('right_hand', 'SALMONQUAN')),
     (6.6408620938365295e-26, ('unfavCable', 'SLICEDCHEESETYPE')),
     (6.6781099289295654e-26, ('atheist', 'DT_MAGN')),
     (6.7105413041097542e-26, ('never_smoked', 'DT_MAGN')),
     (6.731227852206363e-26, ('currently_smoke', 'RAWTOMATOESQUAN'))]




```python
As you can see, a lot of 
```

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
