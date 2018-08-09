
## Expected Loss
Let us return to our discussion of the constant model with the example of tip percentages from the previous chapters. Suppose we only have data from a simple random sample of tips for all waiters at a particular restaurant in a 3-month period, and we want to predict a single tip percentage for all tips from this restaurant. To generalize a model on our 3-month sample to a model on the entire tip history of the restaurant, we must assume that tipping at this restaurant is the same year-round. Since we only have access to data from the sample, we can find the model that empirically minimizes the loss for the sample data with the mean absolute, mean squared, and Huber loss functions. However, if we want to generalize our model, we would need another loss function that does not specifically depend on which data points appear in our sample. The *expected loss* is an example of such a loss function that allows us to analyze the relationship between our sample and the population.

## Definition
We define the population as all tip percentages that have ever been observed at a restaurant, and we define the random variable $ X $ as the percentage tipped for a table in the population. As before, we use $ \theta^* $ to represent our model for the universal tip percentage in the population. Using the quadratic loss as an example, we obtain the following loss function for the expected squared loss:

\\[ f(\theta) =  \mathbb{E}[(X - \theta)^2] \\]

## Minimizing the Expected Squared Loss
Note that the expected squared loss is a function of $ \theta $ since $ \theta $ is a model parameter that we get to choose. We can rewrite $X - \theta$ as $X - \mathbb{E}[X] + \mathbb{E}[X] - \theta$ and simplify using the linearity of expectation to find the value of $ \theta $ that minimizes the expected squared loss.

\\[ \begin{aligned}
f(\theta) &=  \mathbb{E}[(X - \theta)^2] \\\\
&= \mathbb{E}[(X - \mathbb{E}[X] + \mathbb{E}[X] - \theta)^2] \\\\
&= \mathbb{E}[(X - \mathbb{E}[X])^2 + 2(X - \mathbb{E}[X])(\mathbb{E}[X] - \theta) + (\mathbb{E}[X]- \theta)^2] \\\\
&= \mathbb{E}[(X - \mathbb{E}[X])^2] + \mathbb{E}[2(X - \mathbb{E}[X])(\mathbb{E}[X] - \theta)] + \mathbb{E}[(\mathbb{E}[X]- \theta)^2] \\\\
&= \mathbb{E}[(X - \mathbb{E}[X])^2] + 2(\mathbb{E}[X] - \theta)\mathbb{E}[X - \mathbb{E}[X]] + (\mathbb{E}[X]- \theta)^2 \\\\
\end{aligned} \\]

The second term in this expression simplifies to $ 2(\mathbb{E}[X] - \theta)(\mathbb{E}[X] - \mathbb{E}[X]) $, which is simply $ 0 $. Thus, we are left with:

\\[ f(\theta) = \mathbb{E}[(X - \mathbb{E}[X])^2] + (\mathbb{E}[X]- \theta)^2 \\]

Note that the first term of the expression does not depend on $\theta$, so we only need to find the value of $\theta$ that minimizes the second term to minimize the entire expression. Since the second term is a quadratic, we know that $ \hat{\theta} = \mathbb{E}[X] $ minimizes the term and hence the entire expression. If we plug in this value back into the original loss function, we get:

\\[ f(\mathbb{E}[X]) = \mathbb{E}[(X - \mathbb{E}[X])^2] \\]

Notice that this is simply the definition of $ Var(X) $, which means that $ Var(X) $ is the minimized error of the expected squared loss. Consequently,
if $ Var(X) = 0 $, there is no spread in the population, so $ \hat{\theta} = \mathbb{E}[X] $ perfectly predicts every point in the population with no error. On the other hand, if $ Var(X) $ is high, many data points in the population will be far away from $ \mathbb{E}[X] $, resulting in a large error.

## Comparison with MSE
Ideally, we would like to set $ \hat{\theta} = \mathbb{E}[X] $, but we must be able to calculate $ \mathbb{E}[X] $ to do this. If we let $ \mathbb{X} $ be the set of all possible tip percentages in the population, then we can use the definition of the expected value to expand $ \mathbb{E}[X] $.

\\[ \mathbb{E}[X] = \sum_{x \in \mathbb{X}} x \cdot P(X = x) \\]

Unfortunately, we cannot find $ P(X = x) $, the probability that a specific tip percentage appears in the population, because we do not have access to data from the population. Since the empirical distribution of a simple random sample from the population will resemble the distribution of the population, we can pretend that our sample is the population. This allows us to estimate $ \mathbb{E}[X] $ with the mean of our sample since each point appears in our sample with probability $ \frac{1}{n} $ if our sample has size $ n $.

\\[
\mathbb{E}[X] \approx \frac{1}{n} \sum_{i=1}^n x_i
\\]

If we have data from a sample, we can always minimize the mean squared loss of a constant model on the sample by setting $ \theta $ equal to the mean of the dataset. However, if we want to generalize our model to new points in the population, our sample must be a large random sample from the population so that the mean of the sample is a good estimate of $ \mathbb{E}[X] $ for the population. Thus, a model that minimizes the squared loss on a large random sample will also have an error that is close to the minimal squared loss of the population.

## Summary

In this section, we introduced the expected loss as an example of a loss function that helps us generalize from a sample to the population. Often, we only have access to data from a large random sample, so using loss functions such as the MSE gives us a good approximation of the expected loss for the population.

