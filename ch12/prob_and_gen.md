# Probability and Generalization

So far we have used datasets for estimation and prediction by defining a model, selecting a loss function, and minimizing the loss function for the dataset we have on hand. For example, we defined a constant model and the mean squared error loss function for the tips dataset and found that the mean tip percentage of the dataset minimized the MSE.

Even though we now have models that reflect our data, we have no way of generalizing our observations to the larger population. To address the issue of generalization, we will first introduce random variables as a way of representing population data, and we will use expectation and variance to summarize a random variable's distribution. Then, we will introduce the expected loss as a method of analyzing the relationship between a population and sample.
