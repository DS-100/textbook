
## Multiclass Classification

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
