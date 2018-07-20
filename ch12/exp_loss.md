## Expected Loss
Let us return to our discussion of the constant model with the tip percentage dataset in the previous chapters. Suppose we only have data from a simple random sample of tips for all waiters at a particular restaurant in a 3-month period, and we want to predict a single tip percentage for this restaurant. To generalize a model on our 3-month sample to a model on the entire history of the restaurant, we must assume that tipping at this restaurant is the same year-round. Since we only have access to data from the sample, we can find the model that empirically minimizes the loss for the sample data with the mean absolute, mean squared, and Huber loss functions. However, if we want to generalize our model, we would need another loss function that does not specifically depend on which data points appear in our sample. The *expected loss* is an example of such a loss function that allows us to analyze the relationship between our sample and the population.

## Definition
We define the population as all tip percentages that have ever been observed at the restaurant, and we define the random variable $ X $ as the percentage tipped for a table in the population. As before, we use the constant $ \theta $ to represent our model for the universal tip percentage. Using the quadratic loss as an example, we obtain the following loss function:

\\[ f(\theta) =  \mathbb{E}[(X - \theta)^2] \\]

## Minimizing the Expected Squared Loss
Note that the expected squared loss is a function of $ \theta $ since $ \theta $ is a model parameter that we get to choose. We can rewrite $X - \theta$ as $X - \mathbb{E}[X] + \mathbb{E}[X]$ and simplify using the linearity of expectation to find the value of $ \theta $ that minimizes the expected squared loss.

\\[ \begin{aligned}
f(\theta) &=  \mathbb{E}[(X - \theta)^2] \\\\
&= \mathbb{E}[X^2 - 2\theta X + \theta^2] \\\\
&= \mathbb{E}[X^2] - \mathbb{E}[2\theta X] + \mathbb{E}[\theta^2] \\\\
&= \mathbb{E}[X^2] - 2\theta \mathbb{E}[X] + \theta^2 \\\\
\end{aligned} \\]

To find the value of $ \theta $ that minimizes this function, we take the derivative of the final expression with respect to $ \theta $.

\\[ \frac{\partial}{\partial \theta}f(\theta) = -2\mathbb{E}[X] + 2\theta \\]

Now, we can find the value of $ \theta $ where the derivative is zero.

\\[ \begin{aligned}
-2\mathbb{E}[X] + 2\theta &= 0 \\\\
2\theta &= 2\mathbb{E}[X] \\\\
\theta &= \mathbb{E}[X] \\\\
\end{aligned} \\]

Thus, $ \theta = \mathbb{E}[X] $ minimizes the expected squared loss. If we plug in this value back into the original loss function, we get:

\\[ f(\mathbb{E}[X]) = \mathbb{E}[(X - \mathbb{E}[X])^2] \\]

Notice that this is simply the definition of $ Var(X) $, which means that $ Var(X) $ is the minimized error of the expected squared loss. Consequently,
if $ Var(X) = 0 $, there is no spread in the population, so $ \theta = \mathbb{E}[X] $ perfectly predicts every point in the population with no error. On the other hand, if $ Var(X) $ is high, many data points in the population will be far away from $ \mathbb{E}[X] $, resulting in a large error.

## Comparison with MSE
Ideally, we would like to set $ \theta = \mathbb{E}[X] $, but we must be able to calculate $ \mathbb{E}[X] $ to do this. If we let $ \mathbb{X} $ be the set of all possible tip percentages in the population, then we can use the definition of the expected value to expand $ \mathbb{E}[X] $.

\\[ \mathbb{E}[X] = \sum_{x \in \mathbb{X}} x \cdot P(X = x) \\]

Unfortunately, we cannot find $ P(X = x) $, the probability that a specific tip percentage appears in the population, because we do not have access to data from the population. Since a simple random sample from the population will resemble the population, we can pretend that our sample is the population; this allows us to estimate $ \mathbb{E}[X] $ with the mean of our sample, which we will assume has size $ n $.

\\[
\mathbb{E}[X] \approx \frac{1}{n} \sum_{i=1}^n x_i
\\]

If we have data from a sample, we can always minimize the mean squared loss of a constant model on the sample by setting $ \theta $ equal to the mean of the dataset. However, if we want to generalize our model to new points in the population, our sample must be a large random sample from the population so that the mean of the sample is a good estimate of $ \mathbb{E}[X] $ for the population. Thus, a model that minimizes the squared loss on a large random sample will also have an error that is close to the minimal squared loss of the population.

## Summary

In this section, we introduced the expected loss as an example of a loss function that helps us generalize from a sample to the population. Often, we only have access to data from a sample, so using loss functions such as the MSE gives us a good approximation of the expected loss for the population.
