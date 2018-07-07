# Expected Loss
In chapter ten, we introduced the constant model and applied it to the tips dataset with the MSE, absolute, and Huber loss functions. However, this dataset was only a simple random sample of all of the tips a particular waiter received, and we only empirically minimized the loss for this particular sample. If we wanted to generalize from the tips sample to the entire population, we would need another loss function that does not specifically depend on which points appear in our sample. The *expected loss* is an example of such a function that allows us to analyze the relationship between the sample and the population.

## Definition
We use the random variable $ X $ to represent the percentage tipped for a particular bill in the population and the constant $ c $ to represent our predicted value for the tip percentage. Using the quadratic loss as an example, we get the following loss function:

\\[ f(c) =  \mathbb{E}[(X - c)^2] \\]

## Minimizing the Expected Squared Loss
Note that the expected squared loss is a function of $ c $ since we get to choose the predicted value. Ideally, we would like to find the value of $ c $ that minimizes the expected squared loss.


\\[ \begin{aligned}
f(c) &=  \mathbb{E}[(X - c)^2] \\\\
&= \mathbb{E}[X^2 - 2cX + c^2] \\\\
&= \mathbb{E}[X^2] - \mathbb{E}[2cX] + \mathbb{E}[c^2] \\\\
&= \mathbb{E}[X^2] - 2c \mathbb{E}[X] + c^2 \\\\
\end{aligned} \\]

To find the value of $ c $ that minimizes this function, we take the derivative of the final expression with respect to $ c $.

\\[ \frac{\partial}{\partial c}f(c) = -2\mathbb{E}[X] + 2c \\]

Now, we can find the value of $ c $ that makes the derivative zero.

\\[ \begin{aligned}
-2\mathbb{E}[X] + 2c &= 0 \\\\
2c &= 2\mathbb{E}[X] \\\\
c &= \mathbb{E}[X] \\\\
\end{aligned} \\]

Thus, the $c = \mathbb{E}[X]$ minimizes the expected squared loss. If we plug in this value back into the original function, we get:

\\[ f(\mathbb{E}[X]) = \mathbb{E}[(X - \mathbb{E}[X])^2] \\]

Notice that this is simply the definition of $Var(X)$, which means that $Var(X)$ is the minimized error of the expected squared loss.

## Comparison with MSE
As is the case with the tips dataset, we often do not have access to the population data, so we must estimate the expected loss with a sample. If we let $ \mathbb{X} $ be the set of all possible tip percentages in the population, then we can rewrite the expected squared loss as follows:

\\[ \mathbb{E}[(X - c)^2] = \sum_{x \in \mathbb{X}} (x - c)^2 P(X = x) \\]

Unfortunately, we cannot find $ P(X = x) $, the probability that a specific tip percentage appears in the population, since we do not have access to population data. Thus, we must empirically estimate the above expression with only our sample data, which we can use to approximate the population distribution. Then, the probability that a particular point in our dataset occurs is $ \frac{1}{n}$ , and we can estimate the expected squared loss.

\\[
\mathbb{E}[(X - c)^2] \approx \frac{1}{n} \sum_{i=1}^n (x_i - c)^2
\\]

This is the same expression as the MSE error in chapter ten!
