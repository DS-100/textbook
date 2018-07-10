# Expected Loss
Let us return to our discussion of the constant model with the tip percentage dataset in the previous chapters. Recall that we wanted to predict a single tip percentage for all tables using a simple random sample of tips for a particular waiter. Since we only had access to data from this particular sample, we found the model that empirically minimized the loss for data in the sample with the mean absolute, mean squared, and Huber loss functions. If we wanted to generalize from the tips sample to the entire population, however, we would need another loss function that does not specifically depend on which data points appear in our sample. The *expected loss* is an example of such a loss function that allows us to analyze the relationship between our sample and the population.

## Definition
We define the random variable $ X $ as the percentage tipped for a particular table in the population. As before, we use the constant $ \theta $ to represent our model for the universal tip percentage. Using the quadratic loss as an example, we obtain the following loss function:

\\[ f(\theta) =  \mathbb{E}[(X - \theta)^2] \\]

## Minimizing the Expected Squared Loss
Note that the expected squared loss is a function of $ \theta $ since $ \theta $ is a model parameter that we get to choose. After some simplification with the linearity of expectation, we can use calculus to find the value of $ \theta $ that minimizes the expected squared loss.

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

Thus, $\theta = \mathbb{E}[X]$ minimizes the expected squared loss. If we plug in this value back into the original loss function, we get:

\\[ f(\mathbb{E}[X]) = \mathbb{E}[(X - \mathbb{E}[X])^2] \\]

Notice that this is simply the definition of $Var(X)$, which means that $Var(X)$ is the minimized error of the expected squared loss.

## Comparison with MSE
As is the case with the tips dataset, we often do not have access to the population data, so we must estimate the expected loss with data from a sample. If we let $ \mathbb{X} $ be the set of all possible tip percentages in the population, then we can rewrite the expected squared loss as follows:

\\[ \mathbb{E}[(X - \theta)^2] = \sum_{x \in \mathbb{X}} (x - \theta)^2 P(X = x) \\]

Unfortunately, we cannot find $ P(X = x) $, the probability that a specific tip percentage appears in the population, because we do not have access to the population data. Thus, we must use our sample data to approximate the population distribution so that we can estimate the expected loss. If $ n $ is our sample size, then the probability that a particular point appears in our sample is $ \frac{1}{n}$. We can now estimate the expected squared loss as follows:

\\[
\mathbb{E}[(X - c)^2] \approx \frac{1}{n} \sum_{i=1}^n (x_i - c)^2
\\]

This is the same expression as the MSE loss function from the previous chapters; hence, the MSE for a simple random sample provides us with a good estimate of the expected squared loss for the population.

# Summary

In this section, we introduced the expected loss as an example of a loss function that helps us generalize from a sample to the population. Often we only have access to data from a sample, so using loss functions such as the MSE gives us a good approximation of the expected loss for the population.
