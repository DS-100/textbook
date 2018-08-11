
# Statistical Inference

Although data scientists often work with individual samples of data, we are
almost always interested in making generalizations about the population that
the data were collected from. This chapter discusses methods for _statistical
inference_, the process of drawing conclusions about a entire population using
a dataset.

Statistical inference primarily leans on two methods: hypothesis tests and
confidence intervals. In the recent past these methods relied heavily on normal
theory, a branch of statistics that requires substantial assumptions about the
population. Today, the rapid rise of powerful computing resources has
enabled a new class of methods based on _resampling_ that generalize to many
types of populations.

We first review inference using permutation tests and the bootstrap method. We
then introduce bootstrap methods for regression inference and skewed
distributions.
