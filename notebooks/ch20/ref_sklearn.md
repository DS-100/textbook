
# scikit-learn
## Models and Model Selection
| Import | Function        				|  Section  			| Description |  
| ------- | -------- 				|  -------------------------------------------------------------------------------------------------| ---------- | 	------- |
| `sklearn.model_selection` | [`train_test_split(*arrays, test_size=0.2)`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)			| Modeling and Estimation  | Returns two random subsets of each array passed in, with 0.8 of the array in the first subset and 0.2 in the second subset						|  
| `sklearn.linear_model` | [`LinearRegression()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)			| Modeling and Estimation  | Returns an ordinary least squares Linear Regression model			|  
| `sklearn.linear_model` | [`LassoCV()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)			| Modeling and Estimation  | Returns a Lasso (L1 Regularization) linear model with picking the best model by cross validation				|
| `sklearn.linear_model` | [`RidgeCV()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)			| Modeling and Estimation  | Returns a Ridge (L2 Regularization) linear model with picking the best model by cross validation						|
| `sklearn.linear_model` | [`ElasticNetCV()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)			| Modeling and Estimation  | Returns a ElasticNet (L1 and L2 Regularization) linear model with picking the best model by cross validation			|
| `sklearn.linear_model` | [`LogisticRegression()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)			| Modeling and Estimation  | Returns a Logistic Regression classifier						|
| `sklearn.linear_model` | [`LogisticRegressionCV()`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)			| Modeling and Estimation  | Returns a Logistic Regression classifier with picking the best model by cross validation						|

## Working with a Model

Assuming you have a `model` variable that is a `scikit-learn` object:

| Function        				|  Section  			| Description |  
| -------- 				|  -------------------------------------------------------------------------------------------------| ---------- | 	------- |
| `model.fit(X, y)`			| Modeling and Estimation  | Fits the model with the X and y passed in			|
| `model.predict(X)`			| Modeling and Estimation  | Returns predictions on the X passed in according to the model		|
| `model.score(X, y)`			| Modeling and Estimation  | Returns the accuracy of X predictions based on the corect values (y)		|

