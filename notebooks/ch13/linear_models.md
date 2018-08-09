
# Linear Models

Now that we have a general method for fitting a model to a cost function, we
turn our attention to improvements on our model. For the sake of simplicity, we
previously restricted ourselves to a constant model: our model only ever
predicts a single number.

However, giving our waiter such a model would hardly satisfy him. He would
likely point out that he collected much more information about his tables than
simply the tip percents. Why didn't we use his other data—e.g. size of the
table or total bill—in order to make our model more useful?

In this chapter we will introduce linear models which will allow us to make use
of our entire dataset to make predictions. Linear models are not only widely
used in practice but also have rich theoretical underpinnings that will allow
us to understand future tools for modeling. We introduce a simple linear regression model that uses one explanatory variable, explain how gradient descent is used to fit the model, and finally extend the model to incorporate multiple explanatory variables.

