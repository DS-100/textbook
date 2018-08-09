
# Classification

Thus far we have studied models for regression, the process of making
continuous, numerical estimations based on data. We now turn our attention to
**classification**, the process of making categorical predictions based on
data. For example, weather stations are interested in predicting whether
tomorrow will be rainy or not using the weather conditions today.

Together, regression and classification compose the primary approaches for
_supervised learning_, the general task of learning a model based on observed
input-output pairs.

We may reconstruct classification as a type of regression problem. Instead of
creating a model to predict an arbitrary number, we create a model to predict a
probability that a data point belongs to a category. This allows us to reuse
the machinery of linear regression for a regression on probabilities: logistic
regression.

