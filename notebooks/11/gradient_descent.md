
# Gradient Descent and Numerical Optimization

In order to use a dataset for estimation and prediction, we need to precisely
define our model and select a loss function. For example, in the tip percentage
dataset, our model assumed that there was a single tip percentage that does not
vary by table. Then, we decided to use the mean squared error loss function and
found the model that minimized the loss function.

We also found that there are simple expressions that minimize the MSE and the
mean absolute error loss functions: the mean and the median. However, as our models
and loss functions become more complicated we will no longer be able to find
useful algebraic expressions for the models that minimize the loss. For
example, the Huber loss has useful properties but is difficult to differentiate
by hand.

We can use the computer to address this issue using gradient descent, a
computational method of minimizing loss functions.

