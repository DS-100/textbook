

```python
# HIDDEN
import pandas as pd
```

## Multiclass Classification (outline)

Should this be a separate section or just an addendum to an existing section?

- What is multiclass classification
    - More than 2 labels that can be classified; versus a binary classification which is restricted to 2 labels
- A situation where multiclass classification might be used
    - Preferably come up with an example in lebron.csv, but might have to use a different data set
- OvO vs OvR
    - In logistic regression, multiclass classification comes down to a bunch of binary problems
    - OvO: Fit a binary classification problem to each pair of classes, then vote for class that wins the most "head-to-head" classifications
    - OvR/OvA: Fit a binary classification problem for each class versus the rest of the classes; result is class that has the highest probability
- Example walking through multiclass? sklearn's logistic regression supports OvR out of the box.

## Other types of classification problems

- Multilabel: More than one classification problem (e.g. a document can be positive/negative, religion/not religion, political/apolitical, etc.)

## Multiclass Classification

So far we have discussed binary classification, in which our classifier determines whether an observation is part of the positive class or the negative class. However, many data science problems involve **multiclass classification**, in which we would like to classify observations as one of several different classes. For example, we may be interested in knowing whether an animal is a dog, a cat, a squirrel, or a raccoon. In practice, we use **one-vs-rest (OvR) classification** to solve these types of problems.

### One-Vs-Rest Classification

In OvR classification (also known as one-vs-all, or OvA), we decompose a multiclass classification problem into several different binary classification problems. Returning to the example of animal identification, each observation $X_i$ in the training data would be assigned a label $y_i$ that classifies the animal as either a dog, a cat, a squirrel, or a raccoon. Then for each unique label $k$, we construct a new label vector $z$ where $z=1$ if $y=k$ and $z=0$ if $y \neq k$. With our four new label vectors, we can then create four separate classification tasks that predicts whether an observation belongs to the positive class. 

So far in binary classification, we have used the output of these predictors - for example, we were primarily interested in whether the observation was a dog or not a dog. However, in multiclass classification we use the probabilities that each classifier outputs, then we assign the label $k$ for which the respective classifier outputs the highest probability score. Suppose the four classifiers output the probability scores shown below:


```python
# HIDDEN
probas = pd.DataFrame([[0.75, 0.25], [0.63, 0.37], [0.18, 0.82], [0.44, 0.56]], 
                      index=['Dog', 'Cat', 'Squirrel', 'Raccoon'],
                      columns=['P(y=k)', 'P(y!=k)'])
```


```python
probas
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
      <th>P(y=k)</th>
      <th>P(y!=k)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dog</th>
      <td>0.75</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>Cat</th>
      <td>0.63</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>Squirrel</th>
      <td>0.18</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>Raccoon</th>
      <td>0.44</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>



Since the Dog classifier outputs the highest probability score, we predict that our observation is a dog.

### Iris Dataset


```python
from sklearn.datasets import load_iris

iris = load_iris()
```


```python
print(iris.DESCR)
```

    Iris Plants Database
    ====================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML iris datasets.
    http://archive.ics.uci.edu/ml/datasets/Iris
    
    The famous Iris database, first used by Sir R.A Fisher
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    References
    ----------
       - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...
    


## Multilabel Classification

Another type of classification problem is **multilabel classification**, in which our goal is to solve more than one classification problem for the same dataset. An example would be a document classification system: a document can be classified as having positive or negative sentiments, but it can also be distinguished between religious/nonreligious or political/apolitical. Multilabel problems can also be multiclass; we may want our document classification system to distinguish between a list of genres, or identify the language that the document is written in. 

Because each set of labels in a multiclass classification is mutually exclusive, each problem can be solved independently of the others. We can then combine the outputs so that every observation is assigned a set of classification labels.

## Summary

(to-do)
