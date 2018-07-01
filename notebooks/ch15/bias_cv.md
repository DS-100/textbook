
## Cross-Validation

In the previous section, we observe that the we need a way to simulate test error to more accurately manage the bias-variance trade off. Cross-validation provides a method of estimating our model error using a single observed dataset by separating data used for training from the data used for model selection and final accuracy. 

## K-Fold Cross-Validation

In Chapter 14.3, we mentioned the method of the **train-validation split**, which can be used to simulate test error through the validation set. However, this method is prone to high variance of the validation error because the evaluation of the error may depend heavily on which points end up in the training and validation sets.

To tackle this problem, we can run the train-validation split multiple times on the same dataset. The dataset is divided into *k* equally-sized subsets subsets (*$k$ folds*), and the train-validation split is repeated k times. Each time, one of the *k* folds is used as the validation set, and the reaming *k-1* folds are used as the training set. We report the model's final validation error as the average of the $ k $ validation errors from each trial. This method is called **k-fold cross-validation**. The biggest advantage of this method is that every data point is used for validation exactly once, and for training *k-1* times.

The diagram below illustrates the technique when using five folds:

![feature_5_fold_cv.jpg](https://github.com/DS-100/textbook/blob/master/assets/feature_5_fold_cv.jpg?raw=true)

The biggest advantage of this method is that every data point is used for validation exactly once and for training *k-1* times. Typically, a *k* between 5 to 10 is used, but *k* remains an unfixed parameter. When *k* is small, the error estimate has a lower variance (many validation points) but has a higher bias (fewer training points). Vice versa, with large *k* the error estimate has lower bias but has higher variance. 

$k$-fold cross-validation takes more computation time since we typically have to refit each model from scratch for each fold. However, it computes a more accurate validation error by averaging multiple errors together for each model.

The `scikit-learn` library provides a convenient [`sklearn.model_selection.KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) class to implement $k$-fold cross-validation.

## Bias-Variance Tradeoff

K-fold cross-validation helps us manage the bias-variance tradeoff more accurately. Intuitively, the validation error estimates test error by checking the model's performance on a dataset not used for training, allowing us to estimate both model bias and model variance. K-fold cross-validation also allows us to incorporate the fact that the noise in the training set only affects the noise term in the risk, whereas the noise in the training set only affects bias and model variance. To choose the final model to use, we select the one that has the lowest validation error.



## Summary

We use the widely useful cross-validation technique to manage the bias-variance tradeoff. After computing a train-validation-test split on the original dataset, we use the following procedure to train and choose a model.

1. For each potential set of features, fit a model using the training set. The error of a model on the training set is its *training error*.
1. Check the error of each model on the validation set using $k$-fold cross-validation: its *validation error*. Select the model that achieves the lowest validation error. This is the final choice of features and model.
1. Calculate the *test error*, error of the final model on the test set. This is the final reported accuracy of the model. We are forbidden from adjusting the model to increase test error; doing so effectively converts the test set into a validation set. Instead, we must collect a new test set after making further changes to the model.
